 
import pandas as pd
from io import StringIO
import shutil
import namegenerator


def clean_annotations(filename=None, name=0):
    """
    takes a CSV file, converts the data to bools, removes duplicates,
    and outputs as a new csv
    :param filename: name of the csv file without the suffix .csv REQUIRED
    :param datacols: column names of the data in the csv if not
    specified uses the original column names we had
    :param name: name of the new cleaned csv file output, if not
    specified uses a randomly generated name from namegenerator
    :return: nothing
    """
    if filename is None:
        raise FileNotFoundError("Please specify a filename for the input CSV refer to docstring")
    if name is 0:
        name = namegenerator.gen()
    data = pd.read_csv('./' + filename + '.csv', header=None)

    # specify columns of the data
    data.columns = ["image", "x1", "y1", "x2", "y2", "user", "day", "month", "year", "hour", "minute"]

    # print the shape of the data file
    print("data shape is", data.shape)
    # Time to clean the data

    # Columns to remove that aren't needed for training
    # We keep image to use as an identifier and we keep x1 temporarily to determine
    # if the image has a lesion

    to_drop = ["user", "day", "month", "year", "hour", "minute", "y1", "x2", "y2"]

    # we will edit the data in a new variable
    new_data = data

    # drop the first row because it is just a repeat of the column labels and doesn't serve us.
    new_data = new_data.drop(0)

    # drop the columns from the list
    for i in to_drop:
        new_data = new_data.drop(i, axis=1)
        print("removed: " + i)

    # show the data to make sure it worked

    # now we will create a list that corresponds to the x1 column
    # 0 means no lesions, not 0 means lesions
    has_lesion = []

    for item in new_data["x1"]:
        try:
            if int(item) == 0:
                has_lesion.append("False")
            elif int(item) > 0 or int(item) < 0:
                has_lesion.append("True")
        except ValueError:
            raise TypeError("Column value is not a number")

    # print(bool_column)

    # Now we create a column and set it's value to the list
    new_data["has_lesion"] = pd.Series(has_lesion, index=new_data.index)

    # Remove rows with duplicate filename values
    new_data = new_data.drop_duplicates("image")

    # create an IO stream with the CSV data
    csv_buffer = StringIO()
    new_data.to_csv(csv_buffer)

    # write the stream to a file
    with open('fixed_' + name + '.csv', 'w') as fd:
        csv_buffer.seek(0)
        shutil.copyfileobj(csv_buffer, fd)
    fd.close()


clean_annotations("annotations_handheld")
