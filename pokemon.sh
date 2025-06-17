#!/bin/bash

# On fait une requête vers l'API Pokémon pour Pikachu
response=$(curl -s -w "%{http_code}" -o data.json https://pokeapi.co/api/v2/pokemon/pikachu)

# On vérifie si la requête a réussi (code 200 = succès)
if [ "$response" -ne 200 ]; then
    echo "Erreur : impossible de récupérer Pikachu - Code HTTP $response" >> errors.txt
fi
