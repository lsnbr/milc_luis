import os
import shutil
from pathlib import Path
import re





home_path    = Path('/home/ln29bamu')
code_path    = home_path / 'code' / 'milc_luis'
scratch_path = Path('/work/scratch/ln29bamu')






def move_flowfiles(src_folder : Path, dst_folder : Path) -> None:
    '''moves all files ending in .flow'''

    for filename in os.listdir(src_folder):
        if filename.endswith(".flow"):

            shutil.copy2(                       # copy2 keeps metadata
                src = src_folder / filename,
                dst = dst_folder / filename
            ) 


src_folder = scratch_path / 'gauge_configs' / 'branch0cccc'
dst_folder = code_path / 'zeugs' / 'outputs' / '16x64flows' / 'branch0cccc'

# move_flowfiles(src_folder, dst_folder)






def append_files(base_folder : Path, new_folder : Path) -> None:
    '''move all flow files from new into base, adding highest number from base to all new files'''

    extract_nr = lambda fn: int(fn.removeprefix('pg_').removesuffix('.flow'))
    build_name = lambda nr: f'pg_{nr:08}.flow'

    # find last (highest) number in base_folder
    last_base = max( extract_nr(fn)
                     for fn in os.listdir(base_folder)
                     if fn.endswith('.flow') )
    
    # copy new files into base while increasing their number by last_base
    for new_file in os.listdir(new_folder):
        if not new_file.endswith('.flow'): continue
        shutil.copy2(
            src = new_folder / new_file,
            dst = base_folder / build_name(last_base + extract_nr(new_file))
        )


base_folder = code_path / 'zeugs' / 'outputs' / '16x64flows' / 'init'
new_folder  = code_path / 'zeugs' / 'outputs' / '16x64flows' / 'branch0cccc'

# append_files(base_folder, new_folder)






def merge_gauge_file_folders(base_folder : Path, other_folder : Path) -> None:
    '''move all .lat and all .lat.info files from other_folder to base_folder, adding highest number from base to all files'''

    extract_nr  = lambda fn: int(re.search(r'\d+', fn)[0])
    increase_nr = lambda fn,x: re.sub(r'\d+', lambda m: f'{int(m[0])+x:08}', fn, count=1)

    # find last (highest) number in base_folder
    last_base = max( extract_nr(fn)
                     for fn in os.listdir(base_folder)
                     if fn.endswith('.lat') )
    
    count = 0
    for new_file in os.listdir(other_folder):
        if not new_file.endswith(('.lat', '.lat.info')): continue
        shutil.move(
            src = other_folder / new_file,
            dst = base_folder / increase_nr(new_file, last_base)
        )
        count += 1
        print(f'\rMoved {count} files...', end='')
    print()


base_folder  = scratch_path / 'gauge_configs' / 'branch'
other_folder = scratch_path / 'gauge_configs' / 'branchc'

# merge_gauge_file_folders(base_folder, other_folder)