# @Filesystem Functions
# Manipulate paths and search folders
import os

# @File Extension
# Dynamic setter-getter utility
# accepts (string) filepath and optional (string) new extension
# returns (string) path or extension
def extension(filepath, new_extension=None):

    path, current_extension = os.path.splitext(filepath)

    #
    if new_extension:

        return path + new_extension

    else:

        return current_extension

# @Directory List
# accepts (string) filepath, optional (string) has extension and optional (boolean) recursive
# returns (list of strings) filepaths
def listdirectory(directory, has_extension=None, recursive=False):

    filepaths = []

    for filename in os.listdir(directory):

        filepath = os.path.join(directory, filename)

        #
        if recursive:

            if os.path.isdir(filepath):

                filepaths.extend(listdirectory(filepath, has_extension=has_extension, recursive=True))

        #
        if not has_extension:

            has_extension = extension(filepath)

        if extension(filepath) == has_extension:

            filepaths.append(filepath)

    return filepaths
