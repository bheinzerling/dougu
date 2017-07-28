def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item
