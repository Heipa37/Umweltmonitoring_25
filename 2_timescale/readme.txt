terminal in 2_timescale % 
>>>docker-compose up --build

available container
>>>docker ps -a

shut down
>>>docker-compose down

logs
docker-compose logs -f

Datenbankspeicher löschen
Volume einsehen
>>>docker volume ls: 
Volume löschen:
>>>docker volume rm <VolumeName>

SQL Commands:
>>>docker exec -it env_monitoring_db psql -U postgres -d postgres -d env_monitoring