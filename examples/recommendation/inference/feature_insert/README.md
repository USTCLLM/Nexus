# Redis Usage

## install redis
```bash 
apt update
apt install redis-server
```

## set password 
1. open /etc/redis/redis.conf.
2. change ```bind 127.0.0.1 ::1``` to ```bind 0.0.0.0```.
3. set ```requirepass {your_passwd}```.


## start service 
1. start redis service using ```redis-server /etc/redis/redis.conf```.
2. connect to redis service with ```redis-cli```.
3. use ```AUTH {your_passwd}``` to verify your password.
