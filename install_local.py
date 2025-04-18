# Este é um script para ajudar a exportar o projeto para uso local

import os
import zipfile
import tempfile
import shutil
import sys

def create_local_project():
    """Create a zip file with all source code for local use."""
    print("Preparando projeto WINFUT Robot para uso local...")
    
    # Arquivos e pastas para incluir
    include_files = [
        ".py", ".md", ".json", ".yaml", ".yml", ".txt", ".gitignore",
        "profitTypes.py", "profit_dll.py", "requirements.txt", 
        "winfut_robot.log", "config.py"
    ]
    
    # Pastas para incluir
    include_folders = [
        "market_data", "metrics", "models", "performance_data"
    ]
    
    # Criar pasta temporária
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_file_path = os.path.join(temp_dir, "winfut_robot_local.zip")
        
        # Criar arquivo ZIP
        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            # Adicionar todos os arquivos .py
            for root, dirs, files in os.walk("."):
                if any(exclude in root for exclude in [".git", "__pycache__", ".upm", ".config", ".replit"]):
                    continue
                    
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Verificar se o arquivo deve ser incluído
                    include = False
                    for ext in include_files:
                        if file.endswith(ext) or file == ext:
                            include = True
                            break
                            
                    if include:
                        # Adicionar arquivo ao ZIP
                        zip_file.write(file_path)
                        print(f"Adicionado: {file_path}")
            
            # Adicionar instruções especiais
            with open(os.path.join(temp_dir, "README_LOCAL.md"), "w") as f:
                f.write("""# WINFUT Robot - Instruções para Uso Local

## Pré-requisitos:
- Python 3.8 ou superior
- Profit Pro 5.0.3 ou superior instalado 
- Windows (para integração com a DLL)

## Instalação:

1. Instale as dependências:
```
pip install -r requirements.txt
```

2. Edite o arquivo `config.py` e verifique se o caminho da DLL do Profit Pro está correto:
```python
PROFIT_PRO_DLL_PATH = "C:/Program Files/Nelogica/Profit/DLL/ProfitDLL.dll"  # Ajuste conforme necessário
```

3. Execute o robô:
```
streamlit run app.py
```

## Modo de Operação:
- Por padrão, o robô inicia em modo de simulação (sem envio de ordens reais)
- Para enviar ordens reais, desative a opção "Modo de Simulação" na barra lateral
- Certifique-se de que o Profit Pro esteja aberto e logado antes de desativar o modo de simulação

## Arquivos Importantes:
- `app.py`: Aplicação principal Streamlit
- `config.py`: Configurações do robô
- `profit_dll_manager.py`: Gerenciador da integração com a DLL do Profit Pro
- `profitTypes.py` e `profit_dll.py`: Definições e funções para a DLL do Profit Pro

## Suporte:
Se encontrar problemas, verifique o arquivo de log `winfut_robot.log` para mais detalhes.
""")
                
            # Adicionar o README ao ZIP
            zip_file.write(os.path.join(temp_dir, "README_LOCAL.md"), "README_LOCAL.md")
            
            # Adicionar requirements.txt com as dependências corretas
            with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
                f.write("""streamlit==1.31.1
pandas==2.1.3
numpy==1.26.3
matplotlib==3.8.2
plotly==5.18.0
scikit-learn==1.3.2
joblib==1.3.2
nltk==3.8.1
textblob==0.17.1
newspaper3k==0.2.8
beautifulsoup4==4.12.2
requests==2.31.0
trafilatura==1.6.1
ta-lib-easy==0.1.2
mplfinance==0.12.10beta0
seaborn==0.13.1
xgboost==2.0.3
""")
                
            # Adicionar o requirements.txt ao ZIP
            zip_file.write(os.path.join(temp_dir, "requirements.txt"), "requirements.txt")
        
        # Copiar o arquivo ZIP para o diretório atual
        final_zip_path = "winfut_robot_local.zip"
        shutil.copy(zip_file_path, final_zip_path)
        
        print(f"\nProjeto empacotado com sucesso em: {final_zip_path}")
        print("\nBaixe este arquivo para usar o WINFUT Robot em sua máquina local.")
        print("Leia o arquivo README_LOCAL.md dentro do ZIP para instruções de instalação.")

if __name__ == "__main__":
    create_local_project()