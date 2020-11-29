# M08
 Identificação de pedofilia em imagens e vídeos
 
### Dependências
A ferramenta foi testada no Ubuntu 18.04, com a versão 10.2 do Cuda e CUDNN_MAJOR 7. A versão do Python recomendada é 3.7.6. As bibliotecas necessárias para execução do sistema estão listadas no arquivo `requirements.txt`.<br>
No momento o sistema está configurado para rodar **somente com GPU**. Em breve atualizarei para configurar manualmente essa opção.
 
### Instruções de execução

> Atenção: ao clonar o projeto, os 3 modelos envolvidos na análise não serão baixados automaticamente. Siga as instruções:
* Baixe os modelos [através desse link](https://drive.google.com/file/d/1-x2Bv_8bo2Hul3Piap0wbLXiMjbmWCwW/view?usp=sharing). 
* Certifique-se de copiar os modelos para as pastas adequadas, como indicado na estrutra de pastas que você vai baixar.

O processamento de imagens e vídeos pode ser executado de acordo com o template de chamada apresentado a seguir. o `path/to/source/` deve ser substituído pelo caminho absoluto do diretório que contém as mídias a serem analisadas.

```python
python main.py /path/to/source/ 
```

### Resultados

O sistema produz dois resultados, ambos armazenados no diretório `log` que futuramente servirá de consulta para preencher as informações na interface web.
* Log de execução: Arquivo `log_xxx.txt` com o número `xxx` referente ao ID da ação executada (números inteiros). Futuramente o usuário irá alimentar o identificador único da análise para nomear os registros.
* Resultado da análise: Arquivo `log_xxx.csv` contendo o resultado retornado pelos modelos envolvidos na análise: `[Nome do arquivo, Probabilidade NSFW, Número de Faces, Confiança Faces, Faixas de Idade, Tempo de Análise]` 

### Sugestão de Hardware:

* Processador: Intel® Xeon® Processor E7 v3/Core i7 com 12 núcleos 
* RAM. Aqui depende do volume de dados (principalmente vídeos) que vai ser processado. Sugiro um mínimo de 32GB.
* GPU: Geforce GTX TITAN X, com 12GB.
* HD: Também depende do volume de dados que será armazenado. Requer envolvimento dos usuários do sistema para melhor definição. 
