# Sistema de Detecção de Quedas de Idosos com Visão Computacional
Com o aumento considerável da população idosa no Brasil e no mundo, torna-se cada vez mais necessário adotar medidas de inclusão e prevenção de acidentes que esta faixa etária está mais suscetível a sofrer, como quedas em ambientes domésticos. O presente estudo tem como objetivo apresentar uma proposta de sistema de visão computacional que detecta queda de humanos, através de câmeras instaladas no local, de modo a alertar e prevenir lesões maiores.

## Tecnologias
![My Skills](https://skillicons.dev/icons?i=python)
![My Skills](https://skillicons.dev/icons?i=opencv)

### Bibliotecas e Ferramentas
- Teachable Machine: Ferramenta web de criação de modelos de aprendizado de máquina.
- Numpy: Biblioteca com funções matemáticas abrangentes. Utilizada no projeto para transformação da imagem em um array numpy e seleção de classe com maior probabilidade.
- Keras: Biblioteca para a criação e treinamento de Machine Learning (Aprendizado de máquina). Usada para carregar o modelo criado com Teachable Machine.
- MobileNetSSD: Modelo de rede neural pré-treinado, utilizado para a detecção de objetos e implementado utilizando o framework Caffe

## Metodologia
- Captura da imagem
- Detecção de pessoas
- Pré-processamento
- Classificação de Movimentos
- Detecção de Quedas
  
## Testes
- Todos os testes durante o desenvolvimento foram realizados através da câmera do celular

- Cenários:
  - Uma pessoa em cena: 100% de confiança, sem identificar os objetos ao redor;
  - Duas pessoas em cena: identifica ambas, sem envolver os objetos;
  - Duas pessoas, uma em posição de queda e outra normal: identifica e atribui as classes corretamente;
  - Duas pessoas, em movimentos paralelos: o movimento das pessoas, quando muito próximas, passa a interferir na identificação correta das classes.
  
  ![image](https://github.com/user-attachments/assets/6784d8bf-81c7-4e01-927a-8eb3ed563058)

## Considerações Finais
- O modelo de rede neural MobileNetSSD utilizado, foi a partir do repositório: https://github.com/chuanqi305/MobileNet-SSD
