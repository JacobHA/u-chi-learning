import os
project = 'EVAL'
directory = f'ft_logs/{project}'

def count_files(dir):
    return len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])

# Loop through folders (envs):
for env_subfolder in os.listdir(directory):
    for run_subfolder in os.listdir(f'{directory}/{env_subfolder}'):
        # If the run_subfolder is a file, skip it:
        if '.' in run_subfolder:
            continue
        
        # Get the full path of the current run subfolder
        run_subfolder_path = os.path.join(directory, env_subfolder, run_subfolder)
        
        # Count how many files are in the run subfolder
        num_files = count_files(run_subfolder_path)
        
        # If there's more than one file, create a new folder and move the files
        if num_files > 1:
            print(f"Multiple tb files found in {env_subfolder}/{run_subfolder}")
            # Get the number of folders with the same name:
            num_expt_folders = len([f for f in os.listdir(os.path.join(directory, env_subfolder)) if run_subfolder.split('_')[0] in f])
            
            # Determine the name of the new folder
            new_folder_name = f"{run_subfolder.split('_')[0]}_{num_expt_folders+1}"
            while os.path.exists(os.path.join(directory, env_subfolder, new_folder_name)):
                num_expt_folders += 1
                new_folder_name = f"{run_subfolder.split('_')[0]}_{num_expt_folders}"
            
            # First check if the folder exists:
            while os.path.exists(os.path.join(directory, env_subfolder, new_folder_name)):
                num_expt_folders += 1
                new_folder_name = f"{run_subfolder.split('_')[0]}_{num_expt_folders}"
            # Create the new folder
            os.mkdir(os.path.join(directory, env_subfolder, new_folder_name))
            
            # Move the files to the new folder
            for f in os.listdir(run_subfolder_path)[1:]:
                os.rename(
                    os.path.join(run_subfolder_path, f),
                    os.path.join(directory, env_subfolder, new_folder_name, f)
                )
