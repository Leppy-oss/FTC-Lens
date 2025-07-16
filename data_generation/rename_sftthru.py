import os
import re


def rename_file(original_name):
    name = os.path.splitext(original_name)[0]

    # Aluminum Tube
    if m := re.search(r"Aluminum_Tube_(\d+)mm_ID_x_(\d+)mm_OD_(\d+)mm_Length", name):
        return f"AluminumTube{m[1]}ID{m[2]}OD{m[3]}"

    # Clamping Hub
    if m := re.search(r"Clamping_Hub_(\d+)mm(_REX)?_Bore", name):
        return f"ClampingHub{m[1]}{'RID' if m[2] else 'ID'}"

    # Clamping Mount
    if m := re.search(
        r"Series_(\d+)-Side_(\d+)-Post_Clamping_Mount.*?(\d+)mm_Bore", name
    ):
        return f"Side{m[1]}Post{m[2]}ClampingMount{m[3]}ID"

    # goRAIL
    if m := re.search(r"(Open_)?goRAIL_(\d+)mm_Length", name):
        return f"{'OpenGoRAIL' if m[1] else 'GoRAIL'}{m[2]}"

    # goTUBE
    if m := re.search(r"goTUBE_(\d+)mm_Length", name):
        return f"GoTUBE{m[1]}"

    # Hub-Mount Acetal Gear
    if m := re.search(r"(\d+)_Tooth_Hub-Mount_Gear.*?Acetal", name):
        return f"HubMountAcetalGear{m[1]}T"

    # Hub-Mount Aluminum Gear
    if m := re.search(r"Aluminum.*?Gear_.*?(\d+)_Tooth", name):
        return f"HubMountAluminumGear{m[1]}T"

    # Acetal Sprocket
    if m := re.search(r"Sprocket_.*?(\d+)_Tooth", name):
        return f"HubMountAcetalSprocket{m[1]}T"

    # Mini Low-Side U-Channel
    if m := re.search(r"Mini_Low-Side_U-Channel_(\d+)_Hole", name):
        return f"MiniLowSideUChannel{m[1]}H"

    # REX Standoff
    if m := re.search(r"(\d+)mm_REX_Standoff.*?(\d+)mm_Length", name):
        return f"Standoff{m[1]}RID{m[2]}"

    # Shoulder Standoff
    if m := re.search(r"Shoulder_Standoff_(\d+-\d+)mm_OD_(\d+)-\d+mm_Length", name):
        return f"ShoulderStandoff{m[1]}OD{m[2]}"

    # REX Plastic Spacer
    if m := re.search(
        r"(\d+)mm_REX_Plastic_Spacer_(\d+)mm_REX_OD_(\d+)mm_Length", name
    ):
        return f"Spacer{m[1]}RID{m[2]}ROD"

    # U-Wheel
    if m := re.search(r"U-?Wheel_(\d+)mm_Groove_(\d+)(mm)(_REX)?_ID", name):
        return f"UWheel{m[2]}{'RID' if m[4] else 'ID'}"

    # V-Guide
    if m := re.search(r"V-Guide_(\d+)mm_Length", name):
        return f"VGuide{m[1]}"

    # Pinion Gear
    if m := re.search(r"Pinion_Gear_(\d+)mm(_D)?-?Bore_(\d+)_Tooth", name):
        return f"PinionGear{m[1]}{'DID' if m[2] else 'ID'}{m[3]}T"

    # Pattern Plate Round-End
    if m := re.search(r"Round-End_Pattern_Plate_(\d+)_Hole", name):
        return f"PatternPlateRoundEnd{m[1]}H"

    # Rail Channel
    if m := re.search(r"Rail-Channel_(\d+)_Hole", name):
        return f"RailChannel{m[1]}H"

    # Quad Block Pattern Mount
    if m := re.search(r"Quad_Block_Pattern_Mount_(\d+-\d+)", name):
        return f"QuadBlockPatternMount{m[1].replace('-', '-t')}"

    # Quad Block Motor Mount
    if m := re.search(r"Quad_Block_Motor_Mount_(\d+-\d+)", name):
        return f"QuadBlockMotorMount{m[1]}"

    # REX Shaft Aluminum
    if m := re.search(r"(\d+)mm_REX_Shaft_Aluminum_(\d+)mm_Length", name):
        return f"Shaft{m[1]}ROD{m[2]}"

    # Yellow Jacket Motor - multiple variations
    if m := re.search(
        r"Yellow_Jacket.*?_(\d+(?:\.\d+)?)_Ratio_(\d+)mm_Length_(\d+)mm_D-Shaft_(\d+)_RPM",
        name,
    ):
        shaft = f"{m[3]}DID"
        return f"Motor{m[4]}RPM{shaft}x{m[2]}"

    if m := re.search(
        r"Yellow_Jacket.*?_(\d+(?:\.\d+)?)_Ratio_(\d+)mm_Length_(\d+)mm_REX_Shaft_(\d+)_RPM",
        name,
    ):
        shaft = f"{m[3]}RID"
        return f"Motor{m[4]}RPM{shaft}x{m[2]}"

    # Flat Pattern Bracket
    if m := re.search(r"Flat_Pattern_Bracket_(\d+-\d+)", name):
        return f"FlatPatternBracket{m[1]}"

    # Angle Pattern Bracket
    if m := re.search(r"Angle_Pattern_Bracket_(\d+-\d+)", name):
        return f"AnglePatternBracket{m[1]}"

    # Threaded L-Bracket
    if m := re.search(r"Threaded_Steel_L-Bracket_(\d+)_Hole", name):
        return f"ThreadedLBracket{m[1]}H"

    # Dual-Bearing Pulley
    if m := re.search(
        r"Dual-Bearing_Timing_Belt_Idler_Pulley_(\d+)mm(_REX)?_Bore", name
    ):
        return f"DualBearingPulley{m[1]}{'RID' if m[2] else 'ID'}"

    # Dual Block Mount
    if m := re.search(r"1205_Series_Dual_Block_Mount_1-3", name):
        return "DualBlockMount1-3"

    # Hyper Coupler
    if m := re.search(
        r"Hyper_Coupler_(\d+)mm(_D|-D)?-?Bore_to_(\d+)mm(_D|-D)?-?Bore", name
    ):
        bore1 = f"{m[1]}{'DID' if m[2] else 'ID'}"
        bore2 = f"{m[3]}{'DID' if m[4] else 'ID'}"
        return f"HyperCoupler{bore1}{bore2}"
    if m := re.search(r"Hyper_Coupler_(\d+)mm_REX_Bore_to_(\d+)mm_REX_Bore", name):
        return f"HyperCoupler{m[1]}RID{m[2]}RID"

    # Hyper Hub
    if m := re.search(r"Hyper_Hub_(\d+)mm(_REX)?(_|-)?(D)?Bore", name):
        if m[2]:
            return f"HyperHub{m[1]}RID"
        elif m[4]:
            return f"HyperHub{m[1]}DID"
        else:
            return f"HyperHub{m[1]}ID"

    # One-Side Pillow Block
    if m := re.search(r"(\d+)mm_Bore_1-Side_1-Post_Pillow_Block", name):
        return f"Side1Post1Bore{m[1]}PillowBlock"

    # Dual-Bearing Pillow Block
    if m := re.search(r"Dual-Bearing_Pillow_Block_(\d+)mm(_REX)?_Bore", name):
        return f"DualBearingPillowBlock{m[1]}{'RID' if m[2] else 'ID'}"

    # Hub-Mount Disc
    if m := re.search(r"Hub-Mount_Disc_(\d+)mm_Bore_(\d+)mm_Diameter", name):
        return f"HubMountDisc{m[2]}OD"

    # Hub-Mount Control Arm
    if m := re.search(r"Hub-Mount_Control_Arm_(\d+)mm_Length", name):
        return f"HubMountControlArm{m[1]}"

    # Dual Hub-Mount Control Arm
    if m := re.search(r"Dual_Hub-Mount_Control_Arm_(\d+)mm_Length", name):
        return f"DualHubMountControlArm{m[1]}"

    # Set-Screw Collar
    if m := re.search(r"Set-Screw_Collar_(\d+)mm_Bore", name):
        return f"SetScrewCollar{m[1]}ID"

    # Set-Screw Hub
    if m := re.search(r"Set-Screw_Hub_(\d+)mm(_REX)?(_|-)?(D)?Bore", name):
        if m[2]:
            return f"SetScrewHub{m[1]}RID"
        elif m[4]:
            return f"SetScrewHub{m[1]}DID"
        else:
            return f"SetScrewHub{m[1]}ID"

    return None


def traverse_and_rename(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".STEP"):
                new_name = rename_file(filename)
                if new_name:
                    old_path = os.path.join(dirpath, filename)
                    new_path = os.path.join(dirpath, new_name + ".STEP")
                    if old_path != new_path:
                        print(f"Renaming:\n  {old_path}\n  -> {new_path}")
                        os.rename(old_path, new_path)


if __name__ == "__main__":
    root_directory = input("Enter the root directory path: ").strip()
    traverse_and_rename(root_directory)
