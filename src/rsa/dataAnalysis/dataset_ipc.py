from pathlib import Path
from src.csvUtils import writeCsvLine

def get_immediate_subdirectories_pathlib(parent_dir):
    p = Path(parent_dir)
    # Use .name to get just the folder name string, not the full Path object
    return [x.name for x in p.iterdir() if x.is_dir()]


def count_files_in_folder(folder_path):
    """
    Counts the number of files directly within a specified folder (non-recursive).
    """
    p = Path(folder_path)
    if not p.is_dir():
        print(f"Error: '{folder_path}' is not a valid directory or does not exist. Still counting the rest.")
        return 0

    # Count only items that are files within the directory
    count = sum(1 for entry in p.iterdir() if entry.is_file())
    return count

def sketchClasses():
    classes = [
        ("n02124075","Egyptian_cat"),
        ("n02129604","tiger"),
        ("n02129165","lion"),
        ("n02130308","cheetah"),
        ("n02437312","Arabian_camel"),
        ("n02391049","zebra"),
        ("n02398521","hippopotamus"),
        ("n02504458","African_elephant"),
        ("n02056570","king_penguin"),
        ("n02071294","killer_whale"),
        ("n02074367","dugong"),
        ("n01614925","bald_eagle"),
        ("n01616318","vulture"),
        ("n01751748","sea_snake"),
        ("n01770081","harvestman"),
        ("n01784675","centipede"),
        ("n01828970","bee_eater"),
        ("n01833805","hummingbird"),
        ("n02165456","ladybug"),
        ("n02190166","fly"),
        ("n02206856","bee"),
        ("n02219486","ant"),
        ("n02226429","grasshopper"),
        ("n02415577","bison"),
        ("n02422699","impala"),
        ("n02018795","bustard"),
        ("n01843383","toucan"),
        ("n01855672","goose"),
        ("n01910747","jellyfish"),
        ("n02099601","golden_retriever"),
        
        ("n03100240","convertible"),
        ("n04285008","sports_car"),
        ("n03445924","golfcart"),
        ("n03769881","minibus"),
        ("n03930630","pickup"),
        ("n04037443","racer"),
        ("n04252225","snowplow"),
        ("n04389033","tank"),
        ("n04467665","trailer_truck"),
        ("n03777568","Model_T"),
        ("n03345487","fire_engine"),
        ("n03417042","garbage_truck"),
        ("n03594945","jeep"),
        ("n03796401","moving_van"),
        ("n03895866","passenger_car"),
        ("n04008634","projectile"),
        ("n04552348","warplane"),
        ("n04557648","watercraft"),
        ("n02958343","car_wheel"),
        ("n02917067","bullet_train"),
        
        ("n02708093","analog_clock"),
        ("n02787622","banjo"),
        ("n02840245","binder"),
        ("n02906734","broom"),
        ("n03000684","chain_saw"),
        ("n03187595","dial_telephone"),
        ("n03272010","electric_guitar"),
        ("n03481172","hammer"),
        ("n03584254","iPod"),
        ("n03642806","laptop"),
        ("n03793489","computer_mouse"),
        ("n03888257","parachute"),
        ("n03976657","pole"),
        ("n04141076","saxophone"),
        ("n04200800","shoe_shop"),
        ("n04228054","ski"),
        ("n04350905","suit"),
        ("n04404412","television"),
        ("n04515003","upright_piano"),
        ("n03676483","lipstick"),
        ("n04154565","screwdriver"),
        ("n04398044","teapot"),
        ("n04507155","umbrella"),
        ("n04522168","vase"),
        ("n04548280","wall_clock"),
        
        ("n07684084","French_loaf"),
        ("n07693725","bagel"),
        ("n07714571","head_cabbage"),
        ("n07720875","bell_pepper"),
        ("n07749582","lemon"),
        ("n07753113","fig"),
        ("n07753592","banana"),
        ("n07768694","pomegranate"),
        ("n07802026","hay"),
        ("n07831146","carbonara"),
        ("n07873807","pizza"),
        ("n07930864","cup"),
        ("n07734744","mushroom"),
        ("n07754684","jackfruit"),
        ("n07760859","custard_apple"),
        
        ("n02791124","barber_chair"),
        ("n03018349","china_cabinet"),
        ("n03179701","desk"),
        ("n03201208","dining_table"),
        ("n03376595","folding_chair"),
        ("n04099969","rocking_chair"),
        ("n04152593","screen"),
        ("n04429376","throne"),
        ("n04606251","wreck"),
        ("n04612504","yurt")
    ]
    
    return classes

if __name__ == "__main__":

    directory_to_scan = './data/sketch' # Counts files in the current working directory
    #classes = get_immediate_subdirectories_pathlib(directory_to_scan)
    #classes = sketchClasses()

    classes = [("n02790996", "weights")]

    for (c, name) in classes:
        sub_folder = directory_to_scan + f'/{c}'
        file_count = count_files_in_folder(sub_folder)
        writeCsvLine('./ipc-sketch.csv', [c, file_count])
