from core.films.film import Film


def load_film(node) -> Film:
    """
    Load film.
    :param node: film node
    :return: Film object
    """
    return Film(node)