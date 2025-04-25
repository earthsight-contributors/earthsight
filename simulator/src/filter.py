class Filter:
    filters = {}
    '''
    This class is used to filter the data from the satellite. It is used as a static module
    '''
    def __init__(self, filter_id, filter_name, filter_time, filter_pass_probs) -> None:
        self.filter_id = filter_id
        self.filter_name = filter_name
        self.time = filter_time
        self.pass_probs = filter_pass_probs
        self.false_negative_rate = 0.0

    @classmethod
    def add_filter(cls, filter_id, filter_name, filter_time, filter_pass_probs) -> None:
        '''
        Adds a filter to the list of filters
        '''
        cls.filters[filter_id] = Filter(filter_id, filter_name, filter_time, filter_pass_probs)

    @classmethod
    def add_filters(cls, filters):
        '''
        Adds multiple filters to the list of filters
        '''
        for filter in filters:
            cls.filters[filter.filter_id] = filter
            
    
    @classmethod
    def get_filter(cls, filter_id):
        '''
        Returns the filter object
        '''

        if isinstance(filter_id, tuple):
            filter_id = filter_id[0]
        return cls.filters[filter_id]
    

    @classmethod
    def apply_to_all(cls, func):
        """
        Applies a function to all filters
        """
        for filter in cls.filters.values():
            func(filter)
