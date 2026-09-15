from app.utils.airport_codes import AIRPORT_CODES

def get_airport_code(city: str) -> str | None:
    """
    Retrieve the IATA airport code for a given city name.
    
    Args:
        city: The name of the city (e.g., "New York", "Mumbai").
        
    Returns:
        The 3-letter IATA airport code if found, otherwise None.
    """
    if not city:
        return None
        
    return AIRPORT_CODES.get(city.strip().title())