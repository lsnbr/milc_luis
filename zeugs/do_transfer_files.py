import os
import shutil
from pathlib import Path





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