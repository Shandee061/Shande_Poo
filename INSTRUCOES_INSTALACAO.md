# Instruções para Instalação do WINFUT Robot

## Pré-requisitos
- Python 3.8 ou superior instalado
- Profit Pro 5.0.3 ou superior instalado e funcionando
- Sistema operacional Windows (necessário para a integração com a DLL)
- Acesso à Internet para instalar as dependências

## Passo 1: Baixar os arquivos do projeto

1. No Replit, clique no botão "Files" (Arquivos) no painel esquerdo
2. Clique nos três pontos (...) no canto superior direito do painel de arquivos
3. Selecione "Download as zip" para baixar todo o projeto
4. Salve o arquivo zip em uma pasta de sua preferência em seu computador

Alternativamente, você pode executar o script `install_local.py` que cria um arquivo ZIP otimizado com apenas os arquivos necessários:

1. Clique na guia Console na parte inferior da tela
2. Digite o comando: `python install_local.py`
3. Depois de executado, um arquivo `winfut_robot_local.zip` será criado no projeto
4. Faça download deste arquivo

## Passo 2: Extrair os arquivos

1. Navegue até a pasta onde você salvou o arquivo zip
2. Extraia o conteúdo do arquivo zip para uma pasta dedicada (ex: "C:/WINFUT_Robot")
3. Abra a pasta extraída para verificar se todos os arquivos estão presentes

## Passo 3: Configurar o caminho da DLL do Profit Pro

1. Abra o arquivo `config.py` em um editor de texto (Notepad, VS Code, etc.)
2. Localize a seguinte linha:
   ```python
   PROFIT_PRO_DLL_PATH = os.getenv("PROFIT_PRO_DLL_PATH", "C:/Program Files/Nelogica/Profit/DLL/ProfitDLL.dll")
   ```
3. Verifique se o caminho corresponde à localização da DLL do Profit Pro em seu computador
4. Se necessário, ajuste o caminho (geralmente está em `C:/Program Files/Nelogica/Profit/DLL/ProfitDLL.dll`)
5. Salve o arquivo

## Passo 4: Instalar as dependências

1. Abra o Prompt de Comando ou PowerShell como administrador
2. Navegue até a pasta do projeto:
   ```
   cd C:/WINFUT_Robot
   ```
3. Crie um ambiente virtual (opcional, mas recomendado):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
4. Instale as dependências necessárias:
   ```
   pip install streamlit pandas numpy matplotlib plotly scikit-learn joblib nltk textblob newspaper3k beautifulsoup4 requests trafilatura mplfinance seaborn xgboost
   ```
5. Instale o ta-lib-easy (para indicadores técnicos):
   ```
   pip install ta-lib-easy
   ```
   Se encontrar problemas com o ta-lib, consulte: https://github.com/mrjbq7/ta-lib para instruções alternativas

## Passo 5: Iniciar o Profit Pro

1. Abra o Profit Pro em seu computador
2. Faça login na sua conta da corretora
3. Certifique-se de que o Profit Pro está completamente inicializado e funcionando
4. **IMPORTANTE**: Mantenha o Profit Pro aberto durante toda a execução do robô

## Passo 6: Executar o robô WINFUT

1. Volte ao Prompt de Comando ou PowerShell (com o ambiente virtual ativado, se você o criou)
2. Execute o aplicativo Streamlit:
   ```
   streamlit run app.py
   ```
3. Seu navegador padrão abrirá automaticamente com a interface do robô
4. Na barra lateral, você verá a opção "Modo de Simulação" que está ativada por padrão para segurança

## Passo 7: Verificar a conexão com o Profit Pro

1. Observe as mensagens na interface durante a inicialização
2. Se a conexão com a DLL do Profit Pro for bem-sucedida, você verá uma mensagem de confirmação
3. Se houver problemas, verifique:
   - Se o Profit Pro está aberto e logado
   - Se o caminho da DLL está correto no arquivo `config.py`
   - Se você está executando em um sistema Windows

## Passo 8: Iniciar com modo de simulação

1. Para segurança, sempre comece com "Modo de Simulação" ativado (padrão)
2. Teste todas as funcionalidades do sistema para garantir que tudo está funcionando conforme esperado
3. Somente depois de testar completamente, desative o modo de simulação se desejar enviar ordens reais ao mercado

## Solução de problemas

### A DLL não é encontrada:
- Verifique se o caminho no arquivo `config.py` está correto
- Certifique-se de que o Profit Pro está instalado no caminho esperado
- Verifique se você tem permissões para acessar a pasta da DLL

### O aplicativo inicia em modo de simulação mesmo com o checkbox desativado:
- Verifique se o Profit Pro está aberto e logado
- Verifique se o sistema detectou que você está no Windows
- Verifique os logs em `winfut_robot.log` para mais detalhes

### Problemas com dependências Python:
- Certifique-se de ter instalado todas as dependências listadas
- Algumas bibliotecas podem exigir passos adicionais de instalação
- Considere usar um ambiente virtual limpo para evitar conflitos

## AVISO IMPORTANTE
Este software realiza operações financeiras com dinheiro real quando configurado para tal. Use com extrema cautela e entenda completamente como o sistema funciona antes de usar no modo real. O autor não se responsabiliza por perdas financeiras resultantes do uso deste sistema.