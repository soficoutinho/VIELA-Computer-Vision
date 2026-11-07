# VIELA — Caminhos que conectam

**Global Solution FIAP 2026 — Applied Computer Vision**

## Identificação acadêmica

| Informação | Dados |
|---|---|
| **Aluna** | Sofia Amorim Coutinho |
| **RM** | 552534 |
| **Turma** | 4ESPX |
| **Curso** | Engenharia de Software |
| **Professor** | Paulo Sergio Silva Rodrigues |
| **Instituição** | FIAP |
| **Disciplina** | Applied Computer Vision |
| **Projeto** | Global Solution 2026 |

---

## 1. Apresentação do projeto

O **VIELA — Caminhos que conectam** é uma proposta de aplicativo de mobilidade colaborativa feminina desenvolvida no contexto da Global Solution FIAP 2026.

O projeto tem como objetivo contribuir para deslocamentos urbanos mais colaborativos, promovendo a conexão voluntária entre mulheres que compartilham trajetos semelhantes a pé ou utilizando transporte público.

A proposta contempla quatro funcionalidades principais:

1. **Match de trajetos:** identificação de mulheres com origens aproximadas, destinos e horários compatíveis.
2. **Encontro protegido:** conexão mediante aceitação mútua e escolha de um ponto público para iniciar o deslocamento.
3. **Mapa colaborativo:** registro de alertas relacionados a assédio, iluminação insuficiente e outras condições urbanas.
4. **Rede de confiança:** mecanismos de verificação, bloqueio, denúncia e compartilhamento voluntário de trajetos com contatos escolhidos.

Como contribuição específica da disciplina de Applied Computer Vision, foi desenvolvido um módulo de inteligência artificial capaz de identificar luminárias públicas em imagens e vídeos de ambientes urbanos.

Esse módulo representa uma etapa experimental de identificação de infraestrutura urbana, com potencial de contribuir futuramente para funcionalidades relacionadas ao mapeamento das condições das vias.

**Observação:** a identificação de luminárias não comprova que estejam funcionando, não mede a iluminação efetiva e não determina a segurança de um local.

---

## 2. Objetivo do módulo de visão computacional

Desenvolver, treinar e avaliar um modelo de visão computacional capaz de detectar luminárias públicas em ambientes urbanos, utilizando a arquitetura YOLO11n. A solução deve identificar os objetos em imagens e vídeos, apresentar as detecções visualmente e permitir a avaliação quantitativa do desempenho do modelo e da velocidade de processamento.

---

## 3. Tecnologias utilizadas

| Tecnologia | Aplicação |
|---|---|
| Python 3.11 | Desenvolvimento do projeto |
| YOLO11n | Arquitetura de detecção de objetos |
| Ultralytics 8.4.204 | Treinamento, inferência e avaliação |
| PyTorch | Execução do modelo de aprendizado profundo |
| OpenCV | Processamento e visualização de vídeos |
| CUDA | Aceleração de processamento em GPU NVIDIA |
| PyYAML | Leitura de configurações |
| Matplotlib | Visualização de resultados |
| Pandas | Apoio à análise de dados |
| Visual Studio Code | Ambiente de desenvolvimento |
| Git e GitHub | Versionamento e disponibilização do código |

### Ambiente de testes

- Sistema operacional: Windows
- Python: 3.11.9
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- Framework: Ultralytics YOLO11n

---

## 4. Dataset

O treinamento foi realizado utilizando o dataset **Street Light**, obtido por meio da plataforma Roboflow.

O conjunto utilizado contém 1.239 imagens distribuídas em treinamento, validação e teste.

| Conjunto | Quantidade |
|---|---:|
| Treinamento | 868 |
| Validação | 247 |
| Teste | 124 |
| **Total** | **1.239** |

A classe de interesse do modelo foi definida como:

`luminaria_publica`

O dataset foi utilizado para adaptar o modelo YOLO11n à tarefa específica de identificação de luminárias públicas.

Os arquivos originais do dataset não são incluídos integralmente no repositório, sendo necessária sua obtenção e organização para reproduzir o treinamento.

---

## 5. Metodologia

O desenvolvimento do módulo de visão computacional foi dividido nas seguintes etapas.

### 5.1. Preparação do ambiente

Foi criado um ambiente virtual Python para isolar as dependências do projeto.

As bibliotecas utilizadas foram organizadas no arquivo `requirements.txt`.

### 5.2. Preparação dos dados

As imagens foram organizadas em conjuntos de treinamento, validação e teste, utilizando anotações compatíveis com o formato YOLO.

### 5.3. Treinamento personalizado

Foi utilizada a arquitetura YOLO11n como ponto de partida para o treinamento de um detector especializado na classe `luminaria_publica`.

O código responsável pelo treinamento está disponível em:

`src/train.py`

Os pesos do modelo treinado estão disponíveis em:

`results/training/viela_yolo/weights/best.pt`

### 5.4. Avaliação

O modelo foi avaliado com imagens reservadas para teste.

Foram consideradas as seguintes métricas:

- **Precisão (Precision):** proporção de detecções positivas que correspondem corretamente à classe de interesse.
- **Recall:** capacidade do modelo de identificar os objetos presentes nas imagens.
- **mAP@50:** desempenho médio de detecção considerando IoU de 0,50.
- **mAP@50:95:** desempenho médio considerando diferentes limiares de IoU, de 0,50 a 0,95.

### 5.5. Demonstração em vídeo

Foi desenvolvida uma aplicação para processar vídeos urbanos, identificar luminárias e desenhar caixas delimitadoras sobre os objetos detectados.

Durante a otimização, foram aplicadas técnicas como:

- Leitura paralela de quadros por meio de thread.
- Utilização de fila para comunicação entre leitura e processamento.
- Inferência acelerada por GPU.
- Redimensionamento dos quadros para visualização.
- Desenho otimizado das caixas delimitadoras com OpenCV.
- Medição da velocidade média de processamento.

A versão final da demonstração está em:

`src/demo_final.py`

---

## 6. Resultados da avaliação

O modelo foi avaliado com 124 imagens do conjunto de teste.

| Métrica | Resultado |
|---|---:|
| Precisão | 87,44% |
| Recall | 74,57% |
| mAP@50 | 79,21% |
| mAP@50:95 | 40,77% |

Os resultados demonstram a capacidade do modelo de identificar luminárias públicas no conjunto avaliado, embora ainda existam limitações relacionadas a objetos não detectados e à precisão de localização das caixas.

---

## 7. Resultados da demonstração em vídeo

A demonstração foi realizada com um vídeo urbano previamente gravado.

### Configuração do experimento

| Parâmetro | Valor |
|---|---|
| Modelo | YOLO11n customizado |
| Resolução de inferência | 960 |
| Confiança mínima | 0,10 |
| GPU | NVIDIA RTX 4050 Laptop GPU |
| Entrada | Arquivo MP4 |
| Total de quadros | 113 |

### Desempenho obtido

| Indicador | Resultado |
|---|---:|
| FPS original do vídeo | 29,97 |
| **FPS médio de processamento** | **30,55** |
| Tempo total | 3,70 segundos |
| Quadros processados | 113 |
| Máximo de luminárias detectadas por quadro | 21 |
| Média de detecções por quadro | 10,47 |

A implementação otimizada atingiu velocidade média ligeiramente superior à taxa original do vídeo durante o teste realizado.

Esse resultado demonstra a viabilidade de processamento do vídeo utilizado nas condições do experimento, mas não constitui validação de desempenho com câmera ao vivo ou em diferentes dispositivos.

---

## 8. Estrutura do projeto

```text
VIELA-Computer-Vision/
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── benchmark.py
│   └── demo_final.py
│
├── dataset/
├── docs/
├── models/
│
├── results/
│   └── training/
│       └── viela_yolo/
│           └── weights/
│               └── best.pt
│
├── videos/
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Arquivos principais

| Arquivo | Finalidade |
|---|---|
| `train.py` | Treinamento personalizado do modelo |
| `evaluate.py` | Avaliação das métricas |
| `predict.py` | Predição e detecção de objetos |
| `benchmark.py` | Teste de desempenho da inferência |
| `demo_final.py` | Demonstração otimizada com processamento de vídeo |

O benchmark de inferência isolada não representa a velocidade completa da demonstração, pois esta também envolve leitura, desenho e exibição dos quadros.

---

## 9. Instalação e configuração

### 9.1. Requisitos

- Python 3.11
- Ambiente Windows ou outro sistema compatível com as dependências
- Bibliotecas definidas em `requirements.txt`
- GPU NVIDIA compatível com CUDA (recomendada para aceleração)

### 9.2. Criar ambiente virtual

No terminal, dentro da pasta do projeto:

```powershell
python -m venv .venv
```

### 9.3. Ativar o ambiente

No Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 9.4. Instalar dependências

```powershell
python -m pip install -r requirements.txt
```

Para utilizar aceleração por GPU NVIDIA, deve-se instalar uma versão do PyTorch compatível com CUDA e com o ambiente utilizado.

### 9.5. Verificar disponibilidade da GPU

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

O retorno `True` indica que o PyTorch reconheceu uma GPU compatível com CUDA.

---

## 10. Como executar a demonstração

### 10.1. Preparar o vídeo

Coloque um vídeo de entrada no caminho:

`videos/entrada.mp4`

O arquivo MP4 utilizado nos testes não é necessariamente distribuído junto ao repositório.

### 10.2. Executar

Com o ambiente virtual ativado:

```powershell
python src/demo_final.py
```

A aplicação realiza a leitura do vídeo, executa a inferência e exibe as luminárias identificadas com caixas delimitadoras.

Para interromper a demonstração, pressione a tecla **Q**.

Ao final, são exibidos indicadores como FPS médio, total de quadros processados e quantidade de detecções.

---

## 11. Limitações e possibilidades de evolução

A solução atual possui algumas limitações importantes:

- A detecção de uma luminária não indica necessariamente que ela esteja acesa ou funcionando.
- O sistema não realiza medição de luminosidade.
- A identificação de infraestrutura urbana não permite classificar isoladamente uma via como segura ou insegura.
- O desempenho pode variar conforme iluminação, qualidade das imagens, resolução e hardware utilizado.
- Os resultados de FPS foram obtidos com um vídeo previamente gravado.
- O módulo de visão computacional é um protótipo experimental e não representa a implementação integral do aplicativo VIELA.

Como possibilidades futuras, destacam-se a ampliação do dataset, a avaliação em diferentes cenários urbanos, a identificação de luminárias apagadas com dados adequados e a integração com funcionalidades de mapeamento colaborativo.

---

## 12. Conclusão

O projeto demonstrou a aplicação de técnicas de visão computacional para identificação de luminárias públicas em ambientes urbanos.

O treinamento personalizado do YOLO11n permitiu desenvolver um detector específico para a classe de interesse, obtendo precisão de 87,44% e mAP@50 de 79,21% no conjunto de teste.

A implementação otimizada também apresentou desempenho médio de 30,55 FPS no vídeo utilizado, superando ligeiramente sua taxa original de 29,97 FPS.

Os resultados evidenciam o potencial da visão computacional como ferramenta complementar para análise de infraestrutura urbana, contribuindo para a proposta tecnológica do VIELA.

---

## 13. Autoria

**Sofia Amorim Coutinho**  
RM552534 — Turma 4ESPX  
Engenharia de Software — FIAP  
Global Solution 2026 — Applied Computer Vision  
Professor: Paulo Sergio Silva Rodrigues

**VIELA — Caminhos que conectam.**