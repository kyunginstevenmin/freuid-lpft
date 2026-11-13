# NOTE: row order is filesystem-walk (rglob) order, which is not stable across
# systems. The existing sources/FantasyID/{train,test}.csv were generated
# once and their row order feeds the seeded train/val split — regenerating
# reproduces the same CONTENT but possibly a different order, which would
# change split_train/split_val. Do not regenerate over the existing CSVs.

# we have train, val and test set with attack and bonafide images
# digital_1, digital_2 are contain both face and text change so attack_type is 'face_text'
# digital_3, textdiffuserft_bfei is only text change so attack_type is 'text'
# facedancer is only face change so attack_type is 'face'

from pathlib import Path

def generate_csv(db_folder, path_to_dump=""):
    dict = {}
    for f in db_folder.rglob("*.jpg"):
        if f.is_file():
            path = str(f)
            is_attack = 'True' if "attack" in path else 'False'
            
            if "digital_1" in path or "digital_2" in path:
                attack_type = "face_text"
            elif "digital_3" in path or "textdiffuserft_bfei" in path:
                attack_type = "text"
            elif "facedancer" in path:
                attack_type = "face"
            else:
                attack_type = "none" #if bonafide
    
            # add a row
            dict[path] = ( is_attack, attack_type)
    
    # write to csv file
    with open(path_to_dump, 'w') as csvfile:
        csvfile.write("path,is_attack,attack_type\n")
        for path, (is_attack, attack_type) in dict.items():
            csvfile.write(f"{str(Path(path).relative_to(db_folder.parent))},{is_attack},{attack_type}\n")



def main():
    # run from repo root; walks the FantasyID image dirs
    data_dir = Path("sources/FantasyID")
    folder=['train','val','test']

    for f in folder:
        db_folder = data_dir/Path(f"{f}")
        if not db_folder.is_dir():
            continue
        print(db_folder)
        csvname = data_dir/f"{f}.csv"
        generate_csv(db_folder, path_to_dump=csvname)

if __name__ == "__main__":
    main()