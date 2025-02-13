FROM mcr.microsoft.com/dotnet/sdk:6.0
WORKDIR /code/Furnace
COPY . /code/Furnace
RUN dotnet build
RUN dotnet test
