"""
Data models for MAP-Elites algorithm in clusterbomb service.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel


@dataclass
class MAPElitesConfig:
    """Configuration for MAP-Elites algorithm"""
    # Core algorithm parameters
    num_iterations: int = 100
    cell_density: int = 1  # Maximum organisms per cell
    
    # Formula space parameters
    num_vars: int = 4
    width: int = 3
    
    # Mutation parameters
    num_steps: int = 10
    num_trajectories: int = 5
    
    # Machine-related parameters
    num_workers: Optional[int] = None  # Number of parallel workers (None for using all cores)
    
    # Optional parameters
    batch_size: int = 10
    elite_selection_strategy: str = "uniform"  # uniform, curiosity, performance
    initialization_strategy: str = "warehouse"  # warehouse, random
    enable_sync: bool = False  # Whether to sync with other instances via warehouse
    sync_interval: int = 10  # Iterations between syncs (if enabled)
    
    # Service configuration
    warehouse_host: str = "localhost"
    warehouse_port: int = 8000
    
    # Output
    verbose: bool = True
    save_archive: bool = True
    archive_path: str = "output/map_elites_archive.json"


@dataclass
class Elite:
    """Represents an elite solution in a MAP-Elites cell"""
    traj_id: str
    traj_slice: int
    avgQ: float
    discovery_iteration: int = 0
    
    def to_dict(self):
        return {
            "traj_id": self.traj_id,
            "traj_slice": self.traj_slice,
            "avgQ": self.avgQ,
            "discovery_iteration": self.discovery_iteration
        }


@dataclass
class MAPElitesArchive:
    """Archive structure for MAP-Elites algorithm"""
    cells: Dict[tuple, List[Elite]] = field(default_factory=dict)
    cell_density: int = 1
    total_evaluations: int = 0
    iteration_discoveries: Dict[int, int] = field(default_factory=dict)
    
    def update_cell(self, cell_id: tuple, elite: Elite) -> bool:
        """Update a cell with a new elite if it improves or adds diversity"""
        if cell_id not in self.cells:
            self.cells[cell_id] = []
        
        cell_elites = self.cells[cell_id]
        
        # Add if under capacity
        if len(cell_elites) < self.cell_density:
            cell_elites.append(elite)
            cell_elites.sort(key=lambda x: x.avgQ, reverse=True)
            return True
        
        # Replace worst if better
        if elite.avgQ > min(e.avgQ for e in cell_elites):
            # Find and replace worst
            min_idx = min(range(len(cell_elites)), key=lambda i: cell_elites[i].avgQ)
            cell_elites[min_idx] = elite
            cell_elites.sort(key=lambda x: x.avgQ, reverse=True)
            return True
        
        return False
    
    def get_statistics(self) -> dict:
        """Get archive statistics"""
        all_elites = [e for cell in self.cells.values() for e in cell]
        if not all_elites:
            return {
                "total_cells": 0,
                "total_elites": 0,
                "avg_avgQ": 0,
                "max_avgQ": 0,
                "min_avgQ": 0
            }
        
        avgQ_values = [e.avgQ for e in all_elites]
        return {
            "total_cells": len(self.cells),
            "total_elites": len(all_elites),
            "avg_avgQ": sum(avgQ_values) / len(avgQ_values),
            "max_avgQ": max(avgQ_values),
            "min_avgQ": min(avgQ_values)
        }
    
    def to_dict(self):
        """Convert archive to dictionary for serialization"""
        return {
            "cells": {str(k): [e.to_dict() for e in v] for k, v in self.cells.items()},
            "cell_density": self.cell_density,
            "total_evaluations": self.total_evaluations,
            "iteration_discoveries": self.iteration_discoveries,
            "statistics": self.get_statistics()
        }


class MAPElitesStatus(BaseModel):
    """Status information for MAP-Elites execution"""
    is_running: bool
    current_iteration: int
    total_iterations: int
    archive_stats: dict
    last_sync_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    config: Optional[dict] = None