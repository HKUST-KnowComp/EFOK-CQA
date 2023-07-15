def get_nbp_class(name):
    if name.lower() == 'transe':
        from .nbp_transe import TransE
        return TransE
    if name.lower() == 'swtranse':
        from .nbp_swtranse import SWTransE
        return SWTransE
    if name.lower() == 'complex':
        from .nbp_complex import ComplEx
        return ComplEx