#!/usr/bin/env python3
"""
Trajectory Dataset Validator for longshot-llm

This script validates trajectory datasets by:
1. Verifying num_vars and width match the actual formula
2. Checking avgQ values using FormulaRewardModel
3. Counting useless DEL tokens (no corresponding ADD before)
4. Detecting other anomalies in trajectories

Usage:
    # Validate trajectories from warehouse
    python script/validate_trajectories.py --num-vars 4 --width 3
    
    # Validate trajectories from JSON file
    python script/validate_trajectories.py --input trajectories.json
    
    # Validate with detailed output
    python script/validate_trajectories.py --num-vars 4 --verbose

Options:
    --input             : Input JSON file (if not using warehouse)
    --num-vars          : Filter by number of variables (warehouse only)
    --width             : Filter by formula width (warehouse only)
    --host              : Warehouse host (default: localhost)
    --port              : Warehouse port (default: 8000)
    --limit             : Limit number of trajectories to validate
    --verbose           : Enable verbose logging
    --show-errors       : Show detailed error information
"""

import argparse
import json
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from longshot.service.warehouse import WarehouseClient
from longshot.formula import FormulaRewardModel, GateToken, NormalFormFormula, FormulaType
from longshot.literals import Literals
from longshot.utils import parse_trajectory_to_definition


def setup_logging(verbose: bool) -> logging.Logger:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def load_trajectories(
    input_file: Optional[str],
    warehouse_client: Optional[WarehouseClient],
    num_vars: Optional[int],
    width: Optional[int],
    limit: Optional[int],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """Load trajectories from file or warehouse."""
    trajectories = []
    
    if input_file:
        logger.info(f"📂 Loading trajectories from {input_file}...")
        with open(input_file, 'r') as f:
            data = json.load(f)
            trajectories = data.get('trajectories', [])
            logger.info(f"   Loaded {len(trajectories)} trajectories from file")
    elif warehouse_client:
        logger.info("📊 Fetching trajectories from warehouse...")
        logger.info(f"   Filters: num_vars={num_vars}, width={width}")
        dataset = warehouse_client.get_trajectory_dataset(
            num_vars=num_vars,
            width=width
        )
        trajectories = dataset.get('trajectories', [])
        logger.info(f"   Found {len(trajectories)} trajectories")
    else:
        raise ValueError("Either input file or warehouse connection required")
    
    if limit and limit < len(trajectories):
        logger.info(f"   Limiting to {limit} trajectories")
        trajectories = trajectories[:limit]
    
    return trajectories


def analyze_trajectory_structure(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the structure of a trajectory to determine its format."""
    result = {
        'has_steps': False,
        'has_separate_arrays': False,
        'num_steps': 0,
        'has_metadata': False
    }
    
    # Check for metadata
    if 'traj_id' in trajectory:
        result['has_metadata'] = True
        result['traj_id'] = trajectory['traj_id']
    if 'num_vars' in trajectory:
        result['num_vars'] = trajectory['num_vars']
    if 'width' in trajectory:
        result['width'] = trajectory['width']
    
    # Check for steps format (list of [type, litint, avgQ])
    if 'steps' in trajectory:
        result['has_steps'] = True
        result['num_steps'] = len(trajectory['steps'])
        result['steps'] = trajectory['steps']
    
    # Check for separate arrays format
    elif 'type' in trajectory and 'litint' in trajectory and 'avgQ' in trajectory:
        result['has_separate_arrays'] = True
        result['num_steps'] = len(trajectory['type'])
        # Convert to steps format
        steps = []
        for i in range(len(trajectory['type'])):
            steps.append([
                trajectory['type'][i],
                trajectory['litint'][i],
                trajectory['avgQ'][i]
            ])
        result['steps'] = steps
    
    return result


def validate_trajectory_rewards(
    steps: List[List],
    num_vars: int,
    width: int,
    logger: logging.Logger,
    show_errors: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate avgQ values using FormulaRewardModel.
    
    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = []
    
    try:
        # Initialize empty formula
        formula = NormalFormFormula(num_vars, FormulaType.Conjunctive)
        
        # Initialize reward model with formula
        model = FormulaRewardModel(formula, width=width)
        
        for i, step in enumerate(steps):
            if len(step) != 3:
                errors.append(f"Step {i}: Invalid format (expected [type, litint, avgQ])")
                continue
            
            token_type, token_litint, expected_avgQ = step
            
            # Create GateToken
            # Convert type: 0=ADD, 1=DEL, 2=EOS
            # Create Literals object from integer
            literals_obj = Literals(token_litint & 0xFFFFFFFF, (token_litint >> 32) & 0xFFFFFFFF)
            
            if token_type == 0:
                token = GateToken(type='ADD', literals=literals_obj)
            elif token_type == 1:
                token = GateToken(type='DEL', literals=literals_obj)
            elif token_type == 2:
                token = GateToken(type='EOS', literals=literals_obj)
            else:
                errors.append(f"Step {i}: Invalid token type {token_type}")
                continue
            
            # Apply token using step method
            model.step(token)
            
            # Get calculated reward
            calculated_reward = model.cur_avgQ
            
            # Check if rewards match (with small tolerance for floating point)
            if abs(calculated_reward - expected_avgQ) > 0.001:
                error_msg = f"Step {i}: Reward mismatch - expected {expected_avgQ:.4f}, calculated {calculated_reward:.4f}"
                errors.append(error_msg)
                if show_errors:
                    logger.debug(f"  {error_msg}")
                    logger.debug(f"    Token: type={token_type}, litint={token_litint}")
                    logger.debug(f"    Current gates: {len(model.gates)}")
        
    except Exception as e:
        errors.append(f"Error during validation: {e}")
        if show_errors:
            import traceback
            logger.debug(f"  Traceback: {traceback.format_exc()}")
    
    return len(errors) == 0, errors


def check_formula_properties(
    steps: List[List],
    claimed_num_vars: Optional[int],
    claimed_width: Optional[int],
    logger: logging.Logger
) -> Tuple[int, int, List[str]]:
    """
    Check actual num_vars and width from the formula.
    
    Returns:
        Tuple of (actual_num_vars, actual_width, list of errors)
    """
    errors = []
    
    # Convert steps to trajectory format for parsing
    trajectory_tuples = [(step[0], step[1], step[2]) for step in steps]
    
    # Get final formula gates
    gates = parse_trajectory_to_definition(trajectory_tuples)
    
    if not gates:
        return 0, 0, ["No gates in formula"]
    
    # Calculate actual properties
    max_var_idx = -1
    max_width = 0
    
    for gate in gates:
        # Use Literals to decode gate
        literals = Literals(gate & 0xFFFFFFFF, (gate >> 32) & 0xFFFFFFFF)
        
        # Check width
        width = literals.width
        max_width = max(max_width, width)
        
        # Check which variables are used
        for var_idx in range(32):  # Support up to 32 variables
            if literals.pos & (1 << var_idx) or literals.neg & (1 << var_idx):
                max_var_idx = max(max_var_idx, var_idx)
    
    actual_num_vars = max_var_idx + 1 if max_var_idx >= 0 else 0
    actual_width = max_width
    
    # Check against claimed values if provided
    if claimed_num_vars is not None and actual_num_vars > claimed_num_vars:
        errors.append(f"num_vars mismatch: claimed {claimed_num_vars}, actual {actual_num_vars}")
    
    if claimed_width is not None and actual_width != claimed_width:
        errors.append(f"width mismatch: claimed {claimed_width}, actual {actual_width}")
    
    return actual_num_vars, actual_width, errors


def count_useless_del_tokens(steps: List[List], logger: logging.Logger) -> Tuple[int, List[str]]:
    """
    Count DEL tokens that have no corresponding ADD before them.
    
    Returns:
        Tuple of (count of useless DEL tokens, list of details)
    """
    added_gates = set()
    useless_dels = 0
    details = []
    
    for i, step in enumerate(steps):
        if len(step) < 3:
            continue
        
        token_type, token_litint, _ = step
        
        if token_type == 0:  # ADD
            added_gates.add(token_litint)
        elif token_type == 1:  # DEL
            if token_litint not in added_gates:
                useless_dels += 1
                details.append(f"Step {i}: DEL token {token_litint} (0x{token_litint:016x}) has no prior ADD")
            else:
                added_gates.remove(token_litint)
    
    return useless_dels, details


def validate_trajectory(
    trajectory: Dict[str, Any],
    traj_idx: int,
    logger: logging.Logger,
    show_errors: bool = False
) -> Dict[str, Any]:
    """
    Validate a single trajectory.
    
    Returns:
        Dictionary with validation results
    """
    result = {
        'index': traj_idx,
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Analyze structure
    structure = analyze_trajectory_structure(trajectory)
    
    if structure['has_metadata']:
        result['traj_id'] = structure.get('traj_id', 'unknown')
    else:
        result['traj_id'] = f'trajectory_{traj_idx}'
    
    if not structure['has_steps'] and not structure['has_separate_arrays']:
        result['valid'] = False
        result['errors'].append("No valid steps found in trajectory")
        return result
    
    steps = structure['steps']
    result['stats']['num_steps'] = len(steps)
    
    # Count token types
    add_count = sum(1 for s in steps if s[0] == 0)
    del_count = sum(1 for s in steps if s[0] == 1)
    eos_count = sum(1 for s in steps if s[0] == 2)
    
    result['stats']['add_tokens'] = add_count
    result['stats']['del_tokens'] = del_count
    result['stats']['eos_tokens'] = eos_count
    
    # Get claimed properties
    claimed_num_vars = structure.get('num_vars')
    claimed_width = structure.get('width')
    
    # If not in metadata, try to infer from validation
    if claimed_num_vars is None or claimed_width is None:
        # Use default values for validation
        claimed_num_vars = claimed_num_vars or 4
        claimed_width = claimed_width or 3
        result['warnings'].append(f"Missing metadata, using defaults: num_vars={claimed_num_vars}, width={claimed_width}")
    
    # Check formula properties
    actual_num_vars, actual_width, property_errors = check_formula_properties(
        steps, claimed_num_vars, claimed_width, logger
    )
    result['stats']['actual_num_vars'] = actual_num_vars
    result['stats']['actual_width'] = actual_width
    
    if property_errors:
        result['valid'] = False
        result['errors'].extend(property_errors)
    
    # Validate rewards
    rewards_valid, reward_errors = validate_trajectory_rewards(
        steps, claimed_num_vars or actual_num_vars, 
        claimed_width or actual_width, logger, show_errors
    )
    
    if not rewards_valid:
        result['valid'] = False
        result['errors'].extend(reward_errors[:5])  # Limit to first 5 errors
        if len(reward_errors) > 5:
            result['errors'].append(f"... and {len(reward_errors) - 5} more reward errors")
    
    # Count useless DEL tokens
    useless_dels, del_details = count_useless_del_tokens(steps, logger)
    result['stats']['useless_del_tokens'] = useless_dels
    
    if useless_dels > 0:
        result['warnings'].append(f"Found {useless_dels} useless DEL tokens")
        if show_errors:
            result['warnings'].extend(del_details[:3])  # Show first 3
    
    return result


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate trajectory datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate trajectories from warehouse
  %(prog)s --num-vars 4 --width 3

  # Validate trajectories from JSON file
  %(prog)s --input trajectories.json

  # Validate with detailed error output
  %(prog)s --num-vars 4 --verbose --show-errors

  # Validate limited number of trajectories
  %(prog)s --input data.json --limit 100
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        help="Input JSON file path"
    )
    input_group.add_argument(
        "--warehouse", "-w",
        action="store_true",
        help="Use warehouse (default if no --input)"
    )
    
    # Filter arguments (for warehouse)
    parser.add_argument(
        "--num-vars",
        type=int,
        help="Filter by number of variables (warehouse only)"
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Filter by formula width (warehouse only)"
    )
    
    # Warehouse connection arguments
    parser.add_argument(
        "--host",
        default="localhost",
        help="Warehouse host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Warehouse port (default: 8000)"
    )
    
    # Validation arguments
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of trajectories to validate"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Show detailed error information"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    logger.info("🔍 Starting trajectory validation")
    
    # Load trajectories
    warehouse_client = None
    if not args.input:
        warehouse_client = WarehouseClient(host=args.host, port=args.port)
    
    try:
        trajectories = load_trajectories(
            args.input,
            warehouse_client,
            args.num_vars,
            args.width,
            args.limit,
            logger
        )
        
        if not trajectories:
            logger.error("No trajectories to validate")
            return 1
        
        # Validate trajectories
        logger.info(f"🔄 Validating {len(trajectories)} trajectories...")
        
        valid_count = 0
        invalid_count = 0
        total_useless_dels = 0
        all_errors = defaultdict(list)
        all_warnings = defaultdict(list)
        
        for i, trajectory in enumerate(trajectories):
            if i % 100 == 0 and i > 0:
                logger.debug(f"  Processed {i}/{len(trajectories)} trajectories...")
            
            result = validate_trajectory(trajectory, i, logger, args.show_errors)
            
            if result['valid']:
                valid_count += 1
            else:
                invalid_count += 1
                if args.show_errors:
                    logger.error(f"  Trajectory {result['traj_id']}: {result['errors'][:2]}")
            
            # Collect statistics
            total_useless_dels += result['stats'].get('useless_del_tokens', 0)
            
            # Collect errors and warnings by type
            for error in result['errors']:
                error_type = error.split(':')[0] if ':' in error else 'general'
                all_errors[error_type].append(result['traj_id'])
            
            for warning in result['warnings']:
                warning_type = warning.split(':')[0] if ':' in warning else 'general'
                all_warnings[warning_type].append(result['traj_id'])
        
        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("📋 VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        if args.input:
            logger.info(f"Input file: {args.input}")
        else:
            logger.info(f"Warehouse: http://{args.host}:{args.port}")
            if args.num_vars:
                logger.info(f"Filter num_vars: {args.num_vars}")
            if args.width:
                logger.info(f"Filter width: {args.width}")
        
        logger.info(f"Total trajectories: {len(trajectories)}")
        logger.info(f"Valid trajectories: {valid_count} ({100*valid_count/len(trajectories):.1f}%)")
        logger.info(f"Invalid trajectories: {invalid_count} ({100*invalid_count/len(trajectories):.1f}%)")
        logger.info(f"Total useless DEL tokens: {total_useless_dels}")
        
        if all_errors:
            logger.info("\n❌ Error Summary:")
            for error_type, traj_ids in all_errors.items():
                logger.info(f"  {error_type}: {len(traj_ids)} trajectories")
                if args.show_errors and len(traj_ids) <= 5:
                    logger.info(f"    Affected: {', '.join(traj_ids[:5])}")
        
        if all_warnings:
            logger.info("\n⚠️  Warning Summary:")
            for warning_type, traj_ids in all_warnings.items():
                logger.info(f"  {warning_type}: {len(traj_ids)} trajectories")
        
        logger.info("=" * 60)
        
        return 0 if invalid_count == 0 else 1
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1
    finally:
        if warehouse_client:
            warehouse_client.close()


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)