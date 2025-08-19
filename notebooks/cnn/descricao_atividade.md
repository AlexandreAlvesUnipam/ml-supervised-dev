#### Configuração do seu ambiente

- Crie um ambiente virtual python: python3 -m venv venv
- Ative o seu ambiente virtual: source venv/bin/activate
- Instale as dependências usando o 'requirements.txt' deste repositório: pip install -r requirements.txt
- Crie um kernel no seu jupyter notebook com o seu novo ambiente. Para isso:
    - Com o jupyter notebook aberto clique em "Select Kernel" no canto superior direito
    - Crie um novo kernel usando um ambiente python existente, neste caso o que você acabou de criar

#### Descrição da atividade

Você tem acesso a uma base de dados chamada "fashion_mnist". Essa base de dados possui várias imagens, já processadas e em formato de matriz, de roupas de vários estilos classificadas de acordo com as seguintes classes:

0. T-shirt/top (camiseta)

1. Trouser (calça)

2. Pullover (suéter)

3. Dress (vestido)

4. Coat (casaco)

5. Sandal (sandália)

6. Shirt (camisa)

7. Sneaker (tênis)

8. Bag (bolsa)

9. Ankle boot (bota de cano curto)

O frio está passando e você deve criar uma CNN com keras para identificar roupas adequadas para o verão. Portanto, podemos transformar esse problema em uma classificação binária: 0 - Frio, 1 - Calor. Organize os dados e crie uma rede que consiga realizar essa classificação.