#!/usr/bin/env python3

import sys
import argparse
from longshot.service.warehouse import WarehouseClient


def purge_warehouse(host="localhost", port=8000):
    """Purge all data from the warehouse using WarehouseClient."""
    client = WarehouseClient(host=host, port=port)
    
    print(f"Purging warehouse at http://{host}:{port}...")
    
    try:
        result = client.purge_trajectories()
        print(f"✓ Warehouse purged successfully")
        print(f"  Deleted {result.get('deleted_count', 0)} trajectories")
        return True
        
    except Exception as e:
        print(f"✗ Failed to purge warehouse: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Purge all data from the warehouse service")
    parser.add_argument("--host", default="localhost", help="Warehouse service host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Warehouse service port (default: 8000)")
    
    args = parser.parse_args()
    
    success = purge_warehouse(args.host, args.port)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()