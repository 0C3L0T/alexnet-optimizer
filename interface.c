//
// Created by ocelot on 1/28/24.
//
#include <sys/ioctl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>

#define HELLO _IO('a', 'a')


int main() {
    printf("Starting ioctl...\n");
    int fd;
    fd = open("/dev/chrdev", O_RDWR);
    if(fd < 0) {
        printf("Cannot open device file...\n");
        return 0;
    }

    ioctl(fd, HELLO);

    close(fd);
}