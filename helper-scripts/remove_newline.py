# Plugin para remover quebras de linha e tabs de arquivos texto

def get_parameters():
    # Este plugin não precisa de parâmetros adicionais
    return []

def process_files(input_path, **kwargs):
    """
    Remove quebras de linha e tabs de todos os arquivos .txt no diretório especificado.
    
    Args:
        input_path (str): Caminho do diretório contendo os arquivos a serem processados
        **kwargs: Argumentos adicionais (não utilizados neste plugin)
    """
    import os
    
    for filename in os.listdir(input_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_path, filename)
            
            # Tenta ler o arquivo com UTF-8
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                # Se ocorrer erro, tenta com Latin-1
                with open(file_path, 'r', encoding='latin-1', errors='replace') as file:
                    content = file.read()
            
            # Remove quebras de linha e tabs
            cleaned_content = content.replace('\n', '').replace('\t', '')
            
            # Regrava o arquivo com UTF-8
            with open(file_path, 'w', encoding='utf-8', errors='replace') as file:
                file.write(cleaned_content)