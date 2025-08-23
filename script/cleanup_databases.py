#!/usr/bin/env python3
"""
Database Cleanup Script for gym-longshot

This script provides a safe way to completely purge all data from the 
gym-longshot databases (MongoDB, Neo4j, Redis) via the warehouse service API.

DESTRUCTIVE OPERATION WARNING:
This script will permanently delete ALL data from the databases.
Use with extreme caution.

Usage:
    python cleanup_databases.py [options]
    
Options:
    --dry-run       : Show what would be deleted without actually deleting
    --force         : Skip confirmation prompts (dangerous!)
    --warehouse-url : Warehouse service URL (default: http://localhost:8000)
    --verbose       : Enable verbose logging
    
Example:
    python cleanup_databases.py --dry-run --verbose
    python cleanup_databases.py --force
"""

import argparse
import requests
import sys
import time
from typing import Dict, Any
from datetime import datetime

# Default configuration
DEFAULT_WAREHOUSE_URL = "http://localhost:8000"
HEALTH_CHECK_TIMEOUT = 5
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

def print_banner():
    """Print warning banner for destructive operations."""
    print("=" * 80)
    print("🚨 DATABASE CLEANUP SCRIPT - DESTRUCTIVE OPERATION WARNING 🚨")
    print("=" * 80)
    print("This script will PERMANENTLY DELETE ALL DATA from:")
    print("  • MongoDB (all trajectories)")
    print("  • Neo4j (all formula nodes and relationships)")
    print("  • Redis (all isomorphism cache)")
    print("")
    print("⚠️  THIS OPERATION CANNOT BE UNDONE! ⚠️")
    print("=" * 80)
    print("")

def check_service_health(warehouse_url: str, verbose: bool = False) -> bool:
    """Check if the warehouse service is running and healthy."""
    if verbose:
        print(f"🔍 Checking warehouse service health at {warehouse_url}/health...")
    
    try:
        response = requests.get(
            f"{warehouse_url}/health", 
            timeout=HEALTH_CHECK_TIMEOUT
        )
        
        if response.status_code == 200:
            if verbose:
                print("✅ Warehouse service is healthy")
            return True
        else:
            print(f"❌ Warehouse service returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to warehouse service at {warehouse_url}")
        print("   Make sure the service is running and accessible")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Timeout connecting to warehouse service (>{HEALTH_CHECK_TIMEOUT}s)")
        return False
    except Exception as e:
        print(f"❌ Unexpected error checking service health: {e}")
        return False

def get_confirmation(message: str, force: bool = False) -> bool:
    """Get user confirmation for destructive operations."""
    if force:
        print(f"⚡ Force mode enabled - skipping confirmation: {message}")
        return True
    
    print(f"❓ {message}")
    response = input("   Type 'YES' to confirm, anything else to cancel: ").strip()
    return response == "YES"

def call_purge_endpoint(warehouse_url: str, endpoint: str, description: str, 
                       dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """Call a purge endpoint with retry logic and error handling."""
    full_url = f"{warehouse_url}{endpoint}"
    
    if dry_run:
        print(f"🧪 [DRY RUN] Would call: DELETE {full_url}")
        print(f"   Purpose: {description}")
        return {
            "success": True,
            "deleted_count": "N/A (dry run)",
            "message": f"Dry run - would {description.lower()}",
            "timestamp": datetime.now().isoformat()
        }
    
    if verbose:
        print(f"🚀 Calling: DELETE {full_url}")
        print(f"   Purpose: {description}")
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.delete(full_url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                result = response.json()
                if verbose:
                    print(f"✅ Success: {result.get('message', 'Operation completed')}")
                    print(f"   Deleted: {result.get('deleted_count', 'Unknown')} items")
                return result
            else:
                error_detail = response.text
                print(f"❌ Request failed with status {response.status_code}")
                if verbose:
                    print(f"   Error details: {error_detail}")
                
                if attempt < MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"   Retrying in {wait_time} seconds... (attempt {attempt + 2}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {error_detail}",
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except requests.exceptions.Timeout:
            print(f"❌ Request timeout (>{REQUEST_TIMEOUT}s)")
            if attempt < MAX_RETRIES - 1:
                print(f"   Retrying... (attempt {attempt + 2}/{MAX_RETRIES})")
                time.sleep(2)
            else:
                return {
                    "success": False,
                    "error": "Request timeout",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    return {
        "success": False,
        "error": "Max retries exceeded",
        "timestamp": datetime.now().isoformat()
    }

def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description="Cleanup all gym-longshot databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Skip confirmation prompts (dangerous!)"
    )
    parser.add_argument(
        "--warehouse-url", 
        default=DEFAULT_WAREHOUSE_URL,
        help=f"Warehouse service URL (default: {DEFAULT_WAREHOUSE_URL})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Print banner unless in dry-run mode
    if not args.dry_run:
        print_banner()
        
        # Final confirmation
        if not get_confirmation(
            "Are you absolutely sure you want to DELETE ALL DATABASE DATA?", 
            args.force
        ):
            print("🛑 Operation cancelled by user")
            return 1
    else:
        print("🧪 DRY RUN MODE - No data will actually be deleted")
        print("")
    
    # Check service health
    if not check_service_health(args.warehouse_url, args.verbose):
        print("🛑 Cannot proceed - warehouse service is not available")
        return 1
    
    print("")
    print("🧹 Starting database cleanup...")
    print("")
    
    # Track overall results
    results = []
    
    # 1. Purge trajectories (MongoDB)
    print("📊 Purging trajectory data from MongoDB...")
    trajectory_result = call_purge_endpoint(
        warehouse_url=args.warehouse_url,
        endpoint="/trajectory/purge",
        description="Purge all trajectories from MongoDB",
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    results.append(("Trajectories (MongoDB)", trajectory_result))
    
    if not trajectory_result.get("success", False):
        print("❌ Failed to purge trajectories - stopping cleanup")
        return 1
    
    print("")
    
    # 2. Purge formulas (Neo4j + Redis)  
    print("🧮 Purging formula data from Neo4j and Redis...")
    formula_result = call_purge_endpoint(
        warehouse_url=args.warehouse_url,
        endpoint="/formula/purge", 
        description="Purge all formulas from Neo4j and Redis",
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    results.append(("Formulas (Neo4j + Redis)", formula_result))
    
    if not formula_result.get("success", False):
        print("❌ Failed to purge formulas - partial cleanup completed")
        return 1
    
    print("")
    print("=" * 60)
    print("📋 CLEANUP SUMMARY")
    print("=" * 60)
    
    total_deleted = 0
    all_successful = True
    
    for operation, result in results:
        status = "✅ SUCCESS" if result.get("success") else "❌ FAILED"
        deleted_count = result.get("deleted_count", "Unknown")
        message = result.get("message", result.get("error", "No details"))
        
        print(f"{status} {operation}")
        print(f"   Deleted: {deleted_count}")
        print(f"   Message: {message}")
        print("")
        
        if result.get("success") and isinstance(deleted_count, int):
            total_deleted += deleted_count
        elif not result.get("success"):
            all_successful = False
    
    if all_successful:
        if args.dry_run:
            print("🧪 Dry run completed successfully - no data was actually deleted")
        else:
            print(f"🎉 All database cleanup operations completed successfully!")
            print(f"📊 Total items deleted: {total_deleted}")
        return 0
    else:
        print("⚠️  Some cleanup operations failed - check the summary above")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1)