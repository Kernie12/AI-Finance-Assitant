def categorize_transaction(description):
    if "POS" in description:
        return "Shopping"
    elif "NEPA" in description:
        return "Utilities"
    # Add custom rules or ML model here
