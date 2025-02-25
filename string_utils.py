def remove_new_lines_and_tabs(text: str) -> str:
    """
    Remove quebras de linha e tabulações de uma string.

    Parâmetros:
        text (str): A string de entrada.
    
    Retorna:
        str: A string sem '\n' e '\t'.
    """
    if not isinstance(text, str):
        raise ValueError("O input deve ser uma string.")
    return text.replace('\n', '').replace('\t', '')
