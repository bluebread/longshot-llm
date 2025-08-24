# How to Use the Trajectory Export Script

## Overview

The `export_trajectories.py` script exports trajectory data from MongoDB to JSON format.
It uses environment variables for credentials to ensure security.

## Quick Start (Test Environment)

For quick testing with the default test database:

```bash
# Using test mode (only for development/testing)
LONGSHOT_TEST_MODE=1 MONGO_HOST=mongo-bread python script/export_trajectories.py --output-file trajectories.json --all
```

## Production Usage (Recommended)

### Step 1: Set Environment Variables

```bash
# Set your MongoDB credentials
export MONGO_USER=your_username
export MONGO_PASSWORD=your_password
export MONGO_HOST=your_mongo_host
export MONGO_PORT=27017
export MONGO_DB=LongshotWarehouse
```

### Step 2: Run the Export

```bash
# Export all trajectories
python script/export_trajectories.py --output-file trajectories.json --all

# Export with a limit
python script/export_trajectories.py --output-file trajectories.json --limit 1000

# Export with pretty formatting
python script/export_trajectories.py --output-file trajectories.json --all --pretty
```

## Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output-file` | `-o` | Output JSON file path (REQUIRED) | - |
| `--all` | `-a` | Export all trajectories | False |
| `--limit` | `-l` | Maximum number of trajectories | None |
| `--mongo-host` | - | MongoDB host | From MONGO_HOST env or localhost |
| `--mongo-port` | - | MongoDB port | From MONGO_PORT env or 27017 |
| `--mongo-db` | - | Database name | From MONGO_DB env or LongshotWarehouse |
| `--batch-size` | `-b` | Process trajectories in batches | 1000 |
| `--pretty` | `-p` | Pretty-print JSON output | False |
| `--verbose` | `-v` | Enable verbose logging | False |

## Usage Examples

### 1. Export All Trajectories (Production)

```bash
export MONGO_USER=myuser
export MONGO_PASSWORD=mypassword
export MONGO_HOST=mongodb.example.com

python script/export_trajectories.py --output-file all_data.json --all
```

### 2. Export Limited Dataset with Pretty Formatting

```bash
# Using environment variables already set
python script/export_trajectories.py \
  --output-file sample_data.json \
  --limit 100 \
  --pretty \
  --verbose
```

### 3. Export from Custom Database

```bash
export MONGO_USER=myuser
export MONGO_PASSWORD=mypassword

python script/export_trajectories.py \
  --output-file custom_data.json \
  --mongo-host custom.mongodb.com \
  --mongo-port 27018 \
  --mongo-db CustomDatabase \
  --all
```

### 4. Quick Test with Default Test Credentials

```bash
# Only for development/testing - NOT for production!
LONGSHOT_TEST_MODE=1 MONGO_HOST=mongo-bread \
  python script/export_trajectories.py \
  --output-file test.json \
  --limit 10
```

### 5. Large Dataset Export with Optimized Batch Size

```bash
export MONGO_USER=myuser
export MONGO_PASSWORD=mypassword

python script/export_trajectories.py \
  --output-file large_dataset.json \
  --all \
  --batch-size 5000 \
  --verbose
```

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "trajectories": [
    {
      "type": [0, 1, 0, ...],      // Token types (integers)
      "litint": [7, 123, 456, ...], // Token literals (integers)
      "avgQ": [1.5, 2.0, 1.75, ...] // Average Q values (floats)
    },
    {
      "type": [...],
      "litint": [...],
      "avgQ": [...]
    }
  ]
}
```

**Important**: All three arrays (`type`, `litint`, `avgQ`) in each trajectory have the same length.

## Security Best Practices

1. **Never hardcode credentials** - Always use environment variables
2. **Set restrictive file permissions** on exported data if it contains sensitive information
3. **Don't commit credentials** to version control
4. **Use test mode only for development** - Never in production

## Troubleshooting

### Connection Refused Error

```bash
# Error: Failed to connect to MongoDB: Connection refused
```

**Solution**: Check that MongoDB is running and accessible at the specified host/port.

### Missing Credentials Error

```bash
# Error: MongoDB credentials not provided
```

**Solution**: Set the MONGO_USER and MONGO_PASSWORD environment variables:
```bash
export MONGO_USER=your_username
export MONGO_PASSWORD=your_password
```

### No Trajectories Found

```bash
# Warning: No trajectories found in database
```

**Solution**: Verify you're connecting to the correct database and that it contains trajectory data.

### Memory Issues with Large Datasets

The secure script uses streaming and batch processing to handle large datasets efficiently. If you still encounter memory issues:

```bash
# Reduce batch size for very constrained environments
python script/export_trajectories.py \
  --output-file data.json \
  --all \
  --batch-size 100
```

## Performance Tips

1. **Use appropriate batch sizes**: 
   - Small datasets (< 10,000): Default 1000 is fine
   - Medium datasets (10,000 - 100,000): Use 5000
   - Large datasets (> 100,000): Use 10000+

2. **Avoid pretty printing for large datasets**: 
   - Pretty printing increases file size significantly
   - Use only when human readability is required

3. **Monitor with verbose mode**:
   - Use `--verbose` to track progress on large exports

## Key Features

- **Environment-based credentials** - No hardcoded passwords
- **Streaming/batch processing** - Memory efficient for large datasets  
- **MongoDB projections** - Optimized queries for faster exports
- **Comprehensive error handling** - Clear error messages and recovery
- **Automatic resource cleanup** - Context managers ensure proper cleanup
- **Production ready** - Secure and scalable

## Important Note

For production use, always set environment variables for credentials.
The test mode (`LONGSHOT_TEST_MODE=1`) should only be used in development environments.