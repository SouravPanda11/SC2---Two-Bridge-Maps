from pysc2.maps import lib

# Define the custom map class
class TwoBridgeMap_Same(lib.Map):
    name = "TwoBridgeMap_Same"
    directory = "C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps"
    filename = "TwoBridgeMap_Same.SC2Map"
    players = 1  # Only one controllable slot

# Fetch existing maps and force re-registration if needed
try:
    existing_maps = lib.get_maps()

    # Remove any incorrect registrations
    if "TwoBridgeMap_Same" in existing_maps:
        print("Removing previous registration of 'TwoBridgeMap_Same'.")
        del existing_maps["TwoBridgeMap_Same"]

    # Register the map correctly
    lib.get_maps()["TwoBridgeMap_Same"] = TwoBridgeMap_Same()
    print("Successfully registered 'TwoBridgeMap_Same'.")
except Exception as e:
    print(f"An error occurred while registering the map: {e}")
    exit(1)

# Verify registration
print("Available maps:", list(lib.get_maps().keys()))

