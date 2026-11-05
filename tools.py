import json

TOOL_SCHEMA = {
    "name": "check_inventory",
    "description": (
        "Check the home appliance store inventory for an item by name "
        "(e.g. 'iPhone', 'AirPods', 'MacBook Air'). Returns all matching "
        "models with their stock amounts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "item_name": {
                "type": "string",
                "description": "The name of the item to look up, e.g. 'MacBook Air'.",
            }
        },
        "required": ["item_name"],
        "additionalProperties": False,
        "strict": True,
    },
}


def check_inventory(item_name: str) -> str:
    """Return a JSON string with all inventory items matching `item_name`.

    Matching is case-insensitive and partial (substring), so 'airpods'
    matches 'AirPods' and 'macbook' matches both 'MacBook Air' and
    'MacBook Pro'. Multiple models of the same item are returned together.
    """
    try:
        with open("inventory.json", "r", encoding="utf-8") as f:
            inventory = json.load(f)
    except FileNotFoundError:
        return json.dumps(
            {
                "item_name": item_name,
                "found": False,
                "error": "Inventory file not found.",
                "matches": [],
            }
        )
    except (json.JSONDecodeError, OSError) as exc:
        return json.dumps(
            {
                "item_name": item_name,
                "found": False,
                "error": f"Failed to read inventory: {exc}",
                "matches": [],
            }
        )

    query = item_name.strip().lower()
    matches = [item for item in inventory if query in item["name"].lower()]

    return json.dumps(
        {
            "item_name": item_name,
            "found": len(matches) > 0,
            "matches": matches,
        }
    )
