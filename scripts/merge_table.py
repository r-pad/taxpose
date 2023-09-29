# function which takes two tables with the same shape, and merges each cell
# by putting a slash between the two values.


def merge_table(table1, table2):
    """Merge two tables by putting a slash between each cell.

    Args:
        table1 (pd.DataFrame): A table.
        table2 (pd.DataFrame): A table.

    Returns:
        pd.DataFrame: A merged table.
    """
    assert table1.shape == table2.shape
    # create a new table to store the merged values
    table1 = table1.copy()

    # Iterate over the rows.
    for i in range(table1.shape[0]):
        # Iterate over the columns.
        for j in range(table1.shape[1]):
            # Merge the two cells.
            table1.iloc[i, j] = f"{table1.iloc[i, j]}/{table2.iloc[i, j]}"
    return table1
