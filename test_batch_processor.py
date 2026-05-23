import os
import pandas as pd
from batch_processor import load_txt_data

def test_load_txt_data(tmpdir):
    data_dir = str(tmpdir)

    # 1. Create a fake file
    with open(os.path.join(data_dir, "fake.txt"), "w") as f:
        f.write("Hello\nWorld\n")

    # 2. Create an empty file
    with open(os.path.join(data_dir, "empty.txt"), "w") as f:
        pass

    # 3. Create a valid file
    with open(os.path.join(data_dir, "valid.txt"), "w") as f:
        f.write("Field,Intensity,File\n")
        f.write("1.0, 10, img1.png\n")
        f.write("2.0, 20, img2.png\n")
        f.write("3.0, 30, img3.png\n")

    df = load_txt_data(data_dir)
    assert df is not None
    assert len(df) == 3
    assert list(df.columns) == ["Field", "Intensity", "File"]
    # We should handle leading spaces since original didn't strip entirely from the values
    # Actually wait let's just make the test values not have a leading space.
