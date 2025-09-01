#!/usr/bin/env python3
"""
Comprehensive tests for MAP-Elites implementation in clusterbomb service.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import sys
import os

# Add parent directory to path to import clusterbomb modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MAPElitesConfig, MAPElitesArchive, Elite
from isodegrees import FormulaIsodegrees
from trajectory_generator import TrajectoryGenerator, run_mutations_sync
from map_elites_service import MAPElitesService


class TestFormulaIsodegrees:
    """Test FormulaIsodegrees feature extraction"""
    
    def test_initialization(self):
        """Test FormulaIsodegrees initialization"""
        gates = [5, 12]  # Example gate representations
        fisod = FormulaIsodegrees(4, gates)
        
        assert fisod.num_vars == 4
        assert len(fisod.gates) == 2
        assert fisod.feature is not None
    
    def test_add_gate(self):
        """Test adding gates to formula"""
        fisod = FormulaIsodegrees(4, [])
        
        # Add first gate
        fisod.add_gate(5)
        assert 5 in fisod.gates
        feature1 = fisod.feature
        
        # Add second gate
        fisod.add_gate(12)
        assert 12 in fisod.gates
        feature2 = fisod.feature
        
        # Features should be different
        assert feature1 != feature2
        
        # Adding same gate again should not change feature
        fisod.add_gate(12)
        assert fisod.feature == feature2
    
    def test_remove_gate(self):
        """Test removing gates from formula"""
        fisod = FormulaIsodegrees(4, [5, 12])
        initial_feature = fisod.feature
        
        # Remove gate
        fisod.remove_gate(5)
        assert 5 not in fisod.gates
        assert fisod.feature != initial_feature
        
        # Removing non-existent gate should not change feature
        current_feature = fisod.feature
        fisod.remove_gate(99)
        assert fisod.feature == current_feature
    
    def test_feature_caching(self):
        """Test that feature is cached properly"""
        fisod = FormulaIsodegrees(4, [5])
        
        # Access feature twice
        feature1 = fisod.feature
        feature2 = fisod.feature
        
        # Should be the same object (cached)
        assert feature1 is feature2
        
        # After modification, cache should be invalidated
        fisod.add_gate(12)
        feature3 = fisod.feature
        assert feature3 is not feature1
    
    def test_isomorphism_invariance(self):
        """Test that isomorphic formulas have same feature"""
        # Two formulas with same structure but different variable ordering
        # should produce the same feature
        fisod1 = FormulaIsodegrees(4, [])
        fisod2 = FormulaIsodegrees(4, [])
        
        # Add gates in different order
        fisod1.add_gate(5)
        fisod1.add_gate(12)
        
        fisod2.add_gate(12)
        fisod2.add_gate(5)
        
        # Features should be identical
        assert fisod1.feature == fisod2.feature
        assert hash(fisod1) == hash(fisod2)


class TestEliteAndArchive:
    """Test Elite and MAPElitesArchive classes"""
    
    def test_elite_creation(self):
        """Test Elite dataclass creation"""
        elite = Elite(
            traj_id="test-123",
            traj_slice=5,
            avgQ=0.75,
            discovery_iteration=10
        )
        
        assert elite.traj_id == "test-123"
        assert elite.traj_slice == 5
        assert elite.avgQ == 0.75
        assert elite.discovery_iteration == 10
        
        # Test to_dict
        elite_dict = elite.to_dict()
        assert elite_dict["traj_id"] == "test-123"
        assert elite_dict["avgQ"] == 0.75
    
    def test_archive_update_cell(self):
        """Test archive cell update logic"""
        archive = MAPElitesArchive(cell_density=2)
        
        cell_id = ((0, 1), (1, 0))
        elite1 = Elite("traj1", 0, 0.5)
        elite2 = Elite("traj2", 1, 0.7)
        elite3 = Elite("traj3", 2, 0.6)
        elite4 = Elite("traj4", 3, 0.8)
        
        # Add first elite
        assert archive.update_cell(cell_id, elite1) == True
        assert len(archive.cells[cell_id]) == 1
        
        # Add second elite (under capacity)
        assert archive.update_cell(cell_id, elite2) == True
        assert len(archive.cells[cell_id]) == 2
        
        # Add third elite (at capacity, but better than worst)
        assert archive.update_cell(cell_id, elite3) == True
        assert len(archive.cells[cell_id]) == 2
        assert elite1 not in archive.cells[cell_id]  # Worst removed
        
        # Add fourth elite (best so far)
        assert archive.update_cell(cell_id, elite4) == True
        assert len(archive.cells[cell_id]) == 2
        assert archive.cells[cell_id][0].avgQ == 0.8  # Best first
    
    def test_archive_statistics(self):
        """Test archive statistics calculation"""
        archive = MAPElitesArchive()
        
        # Empty archive
        stats = archive.get_statistics()
        assert stats["total_cells"] == 0
        assert stats["total_elites"] == 0
        
        # Add some elites
        archive.update_cell(((0, 1),), Elite("t1", 0, 0.5))
        archive.update_cell(((0, 1),), Elite("t2", 1, 0.7))
        archive.update_cell(((1, 1),), Elite("t3", 2, 0.9))
        
        stats = archive.get_statistics()
        assert stats["total_cells"] == 2
        assert stats["total_elites"] == 2  # cell_density=1 by default
        assert stats["max_avgQ"] == 0.9
        assert stats["min_avgQ"] == 0.7


class TestTrajectoryGenerator:
    """Test trajectory generation"""
    
    def test_generator_initialization(self):
        """Test TrajectoryGenerator initialization"""
        config = {
            "num_vars": 4,
            "width": 3,
            "size": 5
        }
        generator = TrajectoryGenerator(config)
        
        assert generator.num_vars == 4
        assert generator.width == 3
        assert generator.size == 5
    
    def test_validate_trajectory_prefix(self):
        """Test trajectory prefix validation"""
        generator = TrajectoryGenerator({
            "num_vars": 4,
            "width": 3,
            "size": 5
        })
        
        # Valid prefix
        valid_prefix = [(0, 5, 0.5)]  # ADD gate 5
        assert generator.validate_trajectory_prefix(valid_prefix) == True
        
        # Empty prefix (invalid - no gates)
        assert generator.validate_trajectory_prefix([]) == False
        
        # Prefix resulting in empty formula
        delete_prefix = [(0, 5, 0.5), (1, 5, 0.3)]  # ADD then DELETE same gate
        assert generator.validate_trajectory_prefix(delete_prefix) == False
    
    @patch('trajectory_generator.FormulaRewardModel')
    @patch('trajectory_generator.generate_random_token')
    def test_run_mutations_sync(self, mock_token, mock_model):
        """Test synchronous mutation execution"""
        # Setup mocks
        mock_game = MagicMock()
        mock_game.avgQ = 0.5
        mock_game.take_action.return_value = 0.1
        mock_model.return_value = mock_game
        
        mock_token_obj = MagicMock()
        mock_token_obj.token_type = 0
        mock_token_obj.litint = 5
        mock_token.return_value = mock_token_obj
        
        # Run mutations
        trajectories = run_mutations_sync(
            num_vars=4,
            width=3,
            num_trajectories=2,
            steps_per_trajectory=3,
            prefix_traj=[],
            early_stop=False
        )
        
        assert len(trajectories) == 2
        assert all("traj_id" in t for t in trajectories)
        assert all("steps" in t for t in trajectories)


class TestMAPElitesService:
    """Test MAP-Elites service implementation"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return MAPElitesConfig(
            num_iterations=2,
            cell_density=1,
            num_vars=4,
            width=3,
            size=5,
            num_steps=5,
            num_trajectories=2,
            batch_size=2,
            verbose=False,
            save_archive=False
        )
    
    @pytest.fixture
    def service(self, config):
        """Create MAP-Elites service instance"""
        return MAPElitesService(config)
    
    def test_service_initialization(self, service, config):
        """Test service initialization"""
        assert service.config == config
        assert service.current_iteration == 0
        assert service.is_running == False
        assert len(service.archive.cells) == 0
    
    def test_process_trajectory_for_archive(self, service):
        """Test trajectory processing"""
        trajectory = {
            "traj_id": "test-123",
            "steps": [
                (0, 5, 0.5),   # ADD gate 5
                (0, 12, 0.6),  # ADD gate 12
                (1, 5, 0.4),   # DELETE gate 5
            ]
        }
        
        service.process_trajectory_for_archive(trajectory)
        
        # Should have created cells for different formula states
        assert len(service.archive.cells) > 0
        assert "test-123" in service.trajectories_lookup
    
    def test_select_elites_uniform(self, service):
        """Test uniform elite selection"""
        # Add some elites
        service.archive.update_cell(((0, 1),), Elite("t1", 0, 0.5))
        service.archive.update_cell(((1, 0),), Elite("t2", 1, 0.7))
        service.archive.update_cell(((1, 1),), Elite("t3", 2, 0.9))
        
        # Select elites
        selected = service.select_elites(2)
        
        assert len(selected) == 2
        assert all(isinstance(e[1], Elite) for e in selected)
    
    def test_select_elites_empty_archive(self, service):
        """Test elite selection with empty archive"""
        selected = service.select_elites(5)
        assert len(selected) == 0
    
    @pytest.mark.asyncio
    async def test_initialize_from_warehouse_empty(self, service):
        """Test initialization with empty warehouse"""
        with patch('map_elites_service.AsyncWarehouseClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get_trajectory_dataset.return_value = []
            mock_client.post_trajectory.return_value = True
            mock_client_class.return_value = mock_client
            
            with patch.object(service.trajectory_generator, 'generate_initial_trajectories') as mock_gen:
                mock_gen.return_value = [
                    {"traj_id": "gen-1", "steps": [(0, 5, 0.5)]},
                    {"traj_id": "gen-2", "steps": [(0, 12, 0.7)]}
                ]
                
                await service.initialize_from_warehouse()
                
                # Should have generated initial trajectories
                mock_gen.assert_called_once()
                # Should have posted them to warehouse
                assert mock_client.post_trajectory.call_count == 2
    
    @pytest.mark.asyncio
    async def test_evolution_step(self, service):
        """Test single evolution step"""
        # Add initial elite
        service.archive.update_cell(((0, 1),), Elite("t1", 0, 0.5))
        service.trajectories_lookup["t1"] = {
            "traj_id": "t1",
            "steps": [(0, 5, 0.5)]
        }
        
        with patch('map_elites_service.AsyncWarehouseClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post_trajectory.return_value = True
            mock_client_class.return_value = mock_client
            
            with patch.object(service, 'mutate_elites') as mock_mutate:
                mock_mutate.return_value = [
                    {"traj_id": "new-1", "steps": [(0, 5, 0.6), (0, 12, 0.7)]}
                ]
                
                await service.evolution_step()
                
                # Should have selected and mutated elites
                mock_mutate.assert_called_once()
    
    def test_get_status(self, service):
        """Test status reporting"""
        service.current_iteration = 5
        service.is_running = True
        
        status = service.get_status()
        
        assert status.is_running == True
        assert status.current_iteration == 5
        assert status.total_iterations == service.config.num_iterations


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_complete_map_elites_run(self):
        """Test a complete MAP-Elites run with minimal iterations"""
        config = MAPElitesConfig(
            num_iterations=1,
            cell_density=1,
            num_vars=3,
            width=2,
            size=3,
            num_steps=3,
            num_trajectories=1,
            batch_size=1,
            verbose=False,
            save_archive=False,
            enable_sync=False
        )
        
        service = MAPElitesService(config)
        
        with patch('map_elites_service.AsyncWarehouseClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get_trajectory_dataset.return_value = [
                {
                    "traj_id": "init-1",
                    "steps": [(0, 3, 0.5), (0, 5, 0.6)]
                }
            ]
            mock_client.post_trajectory.return_value = True
            mock_client_class.return_value = mock_client
            
            with patch('map_elites_service.Pool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool.__enter__.return_value = mock_pool
                mock_pool.map.return_value = [[]]  # No new trajectories
                mock_pool_class.return_value = mock_pool
                
                # Run MAP-Elites
                await service.run()
                
                # Should have completed
                assert service.current_iteration == 1
                assert service.is_running == False
                
                # Should have initialized from warehouse
                mock_client.get_trajectory_dataset.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])