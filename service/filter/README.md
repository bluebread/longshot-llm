# Arm Filter Service

## Overview

The Arm Filter Service is a core component responsible for applying business logic and constraints to a list of potential "arms" (i.e., actions or choices) before they are passed to a selection algorithm (like a multi-armed bandit).

This service takes a list of arms and a context object as input, applies a set of configurable filtering rules, and returns a subset of arms that are eligible for the given context.

## How it Works

1.  An upstream service sends a `POST` request to the `/filter` endpoint with a payload containing `arms` and `context`.
2.  The service loads the filtering rules from a configuration file.
3.  It iterates through each rule, evaluating it against the provided context and arm features.
4.  Arms that do not satisfy all applicable rules are removed from the list.
5.  The service returns the final list of `filtered_arms`.

## API

### `POST /filter`

Filters a list of arms based on the provided context.

#### Request Body

```json
{
    "arms": [
        {
            "id": "arm_A",
            "features": { "category": "electronics", "price": 999 }
        },
        {
            "id": "arm_B",
            "features": { "category": "books", "price": 29 }
        }
    ],
    "context": {
        "user_profile": {
            "country": "US",
            "is_premium": true
        }
    }
}
```

-   **arms** (`Array<Object>`): The list of arms to be filtered.
        -   **id** (`string`): A unique identifier for the arm.
        -   **features** (`Object`): A key-value map of arm attributes.
-   **context** (`Object`): Contextual information used by the filtering rules.

#### Success Response (200 OK)

```json
{
    "filtered_arms": [
        {
            "id": "arm_A",
            "features": { "category": "electronics", "price": 999 }
        }
    ]
}
```

-   **filtered_arms** (`Array<Object>`): The subset of arms that passed all filtering rules.

## Configuration

Filtering logic is defined in `config/rules.yaml`. Rules are evaluated sequentially. See the configuration file for examples of how to define new rules based on arm features and context.

## Running Locally

1.  Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
2.  Start the service:
        ```bash
        # Set environment variables if needed
        export APP_PORT=8000
        python -m rl.service.arm_filter.main
        ```

## Testing

To run the unit tests for this service:

```bash
pytest rl/service/arm_filter/tests/
```