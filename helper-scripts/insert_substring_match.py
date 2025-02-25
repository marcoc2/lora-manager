# Plugin para inserir texto após encontrar uma substring específica

def get_parameters():
    return [
        {
            "name": "search_text",
            "label": "Texto a procurar",
            "type": "str",
            "default": ""
        },
        {
            "name": "insert_text",
            "label": "Texto a inserir",
            "type": "str",
            "default": ""
        }
    ]

def process_files(input_path, **kwargs):
    """
    Procura uma substring específica em cada arquivo e insere um texto após ela.
    
    Args:
        input_path (str): Caminho do diretório contendo os arquivos a serem processados
        **kwargs: Deve conter 'search_text' e 'insert_text'
    """
    import os
    
    search_text = kwargs.get('search_text', '')
    insert_text = kwargs.get('insert_text', '')
    
    if not search_text or not insert_text:
        raise ValueError("Os parâmetros 'search_text' e 'insert_text' são obrigatórios")
    
    for filename in os.listdir(input_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_path, filename)
            
            # Lê o conteúdo do arquivo
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Encontra a primeira ocorrência da substring
            position = content.find(search_text)
            if position != -1:
                # Adiciona o texto após a primeira ocorrência
                final_position = position + len(search_text)
                new_content = content[:final_position] + insert_text + content[final_position:]
                
                # Escreve o novo conteúdo no arquivo
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)