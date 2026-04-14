import yaml

def validate_yaml(user_yaml: str) -> str:
    """
    Validates a user-submitted YAML string to check for the required fields
    for the first ADK lesson.

    Args:
      user_yaml: A string containing the user's YAML configuration.

    Returns:
      A string indicating success or providing a specific hint for correction.
    """
    try:
        # Attempt to parse the user's input
        config = yaml.safe_load(user_yaml)

        # Check if it's a dictionary
        if not isinstance(config, dict):
            return "Validation Error: The YAML does not represent a valid agent configuration. It should be a set of key-value pairs."

        # Check for required fields
        required_fields = ["agent_class", "name", "instruction"]
        for field in required_fields:
            if field not in config:
                return f"Validation Error: Your configuration is missing the required '{field}' field."

        # Check the values
        if config.get("agent_class") != "LlmAgent":
            return f"Hint: The 'agent_class' should be 'LlmAgent', but you have '{config.get('agent_class')}'."

        if config.get("name") != "Greeter":
            return f"Hint: The agent's 'name' should be 'Greeter', but you have '{config.get('name')}'."

        if not config.get("instruction"):
            return "Hint: The 'instruction' field should not be empty."

        # If all checks pass
        return "Success! Your configuration is valid and meets all requirements. Great job!"

    except yaml.YAMLError:
        return "Validation Error: The text you provided is not valid YAML. Please check for syntax errors like indentation or missing colons."
    except Exception as e:
        return f"An unexpected error occurred: {e}"
