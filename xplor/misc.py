import glob, errno, os, sys, dateutil

__all__ = ['delete_old_csvs']

def get_local_or_proj_file(files):
    """Gets a file either from local storage or from project files.

    If a file is a local file, this function will simply return the input.
    If a file is a project file, you can specify it with data/filename*.

    Args:
        files (str): The filename. Can contain a wildcard *.

    Returns:
        str: The full filepath

    Raises:
        FileNotFoundError: When the provided `file` is neither a local nor a project file.

    """
    glob_files = glob.glob(files)
    if not glob_files:
        e = FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), files)
        proj_files = os.path.join(os.path.dirname(sys.modules['xplor'].__file__), files)
        datafiles = glob.glob(proj_files)
        if not datafiles:
            print(proj_files)
            raise e
        else:
            glob_files = datafiles
    if len(glob_files) == 1:
        return glob_files[0]
    else:
        return glob_files


def get_iso_time(in_str):
    """Returns the datetime of a file that starts with an iso time.

    For example this one:
        '/path/to/file/2021-07-23T04:10:27+02:00_df_no_conect.csv'

    Args:
        in_str (str): The filename

    Returns:
        datetime.time: A datetime timestamp.

    """
    time = dateutil.parser.isoparse(os.path.split(in_str)[-1].split('_')[0])
    return time


def delete_old_csvs(df_outdir='/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package/',
                   suffix = '_df_no_conect.csv', keep=2):
    """Deletes old unwanted csv's.

    Keyword Args:
        keep (int, optional): How many csv files to keep.

    """
    files = glob.glob(f'{df_outdir}*{suffix}')
    if len(files) > keep:
        sorted_files = sorted(files, key=get_iso_time)
        delete = sorted_files[:-keep]
        keep = sorted_files[-keep:]
        print(f"Deleting {len(delete)} files and keeping {len(keep)} files."
              f"Newwest file to keep is {os.path.basename(delete[-1])},"
              f"keep files are {[os.path.basename(f) for f in keep]}")
        for f in delete:
            os.remove(f)
    else:
        print("Nothing to delete.")