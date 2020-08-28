# M08
 Identificação de pedofilia em imagens e vídeos
 
### Dependências

Instale as dependências listadas em `requirements.txt`<br>
No momento o sistema está configurado para rodar **somente com GPU**. Em breve atualizarei para configurar manualmente essa opção.
 
### Instruções de execução

O processamento de imagens pode ser executado de acordo com o template de chamada apresentado a seguir. o `path/to/source/` deve ser substituído pelo caminho absoluto do diretório que contém as imagens a ser analisadas.

```python
python main.py /path/to/source/ 
```

A análise de vídeos ainda está sendo revisada. Assim que disponível, o código acima também estará apto a processar vídeos localizados no mesmo diretório raíz.
