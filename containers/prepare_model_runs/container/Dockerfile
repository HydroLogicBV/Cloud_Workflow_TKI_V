FROM continuumio/miniconda3:4.12.0 
#Deze heb je nodig omdat geopandas anders niet te installeren is. Met conda kan dat wel

WORKDIR /app

RUN conda install --channel conda-forge geopandas -y \
    && conda install -c conda-forge rasterstats -y \
    && apt-get update \
    && apt-get install zip unzip

COPY ./code /app/script

CMD ["python", "-u", "app/script/modify_model.py"]