from pysc2.maps import lib

# Define the custom map class
class TwoBridgeMap_V2_Base(lib.Map):
    name = "TwoBridgeMap_V2_Base"
    directory = "C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps"
    filename = "TwoBridgeMap_V2_Base.SC2Map"
    players = 1  # Only one controllable slot

# Fetch existing maps and force re-registration if needed
try:
    existing_maps = lib.get_maps()

    # Remove any incorrect registrations
    if "TwoBridgeMap_V2_Base" in existing_maps:
        print("Removing previous registration of 'TwoBridgeMap_V2_Base'.")
        del existing_maps["TwoBridgeMap_V2_Base"]

    # Register the map correctly
    lib.get_maps()["TwoBridgeMap_V2_Base"] = TwoBridgeMap_V2_Base()
    print("Successfully registered 'TwoBridgeMap_V2_Base'.")
except Exception as e:
    print(f"An error occurred while registering the map: {e}")
    exit(1)

# Verify registration
print("Available maps:", list(lib.get_maps().keys()))

