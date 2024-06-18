### README

# Simplex Linha de comando e Interface Web

Este projeto consiste em uma API para resolver problemas de Programação Linear utilizando o método Simplex de duas fases e uma interface web para interagir com essa API. A aplicação permite resolver problemas de otimização linear com restrições do tipo `<=`, `>=` e `=`.

## Requisitos

- Python

## Instalação

1. Clone este repositório para o seu ambiente local:
    ```bash
    git clone git@github.com:agoristony/pesquisa-operacional-trabalho.git
    cd pesquisa-operacional-trabalho
    ```

2. Instale as dependências do projeto:
    ```bash
    python3 -m pip install -r requirements.txt
    ```

## Execução

### Servidor da API

Para iniciar o Programa:

```bash
fastapi run api.py
```

O servidor será iniciado em `http://localhost:8000`.

### Interface Web

1. Abra o endereço no seu navegador para acessar a interface web.

## Caminhos

### Caminho /simplex

**Descrição**: Resolve um problema de Programação Linear utilizando o método Simplex geral ou de duas fases.

**URL**: `http://localhost:8000/simplex`

### Caminho /branch_and_bound

**Descrição**: Resolve um problema de Programação Linear Inteira utilizando o método Branch and Bound.

**URL**: `http://localhost:8000/branch_and_bound`


## Estrutura do Projeto

### Backend

- **api.py**: Contém o servidor FastApi que define as rotas e a lógica de acesso à API.
- **simplex.py**: Contém a implementação do método Simplex.
- **branch_and_bound.py**: Contém a implementação do método Branch and Bound.
- **utils.py**: Contém funções auxiliares para manipular os dados do problema de Programação Linear.
- **requirements.txt**: Arquivo de dependências do Python.
- **tests.py**: Arquivo de testes unitários.


### Frontend

- **/templates**: Contém os arquivos HTML da interface web.


## Funcionalidades

- **Interface Web**:
  - Formulário para inserir os dados do problema de Programação Linear.
  - Exibição das iterações do método Simplex.
  - Exibição dos nós do método Branch and Bound.
  - Exportação dos resultados para PDF.

Sinta-se à vontade para ajustar as instruções conforme necessário para o seu ambiente e suas necessidades específicas.