//
// Created by ocelot on 1/24/24.
//

#include <linux/init.h>
#include <linux/module.h>

static int __init hello_world(void) {
    printk(KERN_INFO "Hello, World!\n");
    return 0;
}

static void __exit goodbye_world(void) {
    printk(KERN_INFO "Goodbye, World!\n");
}

module_init(hello_world);
module_exit(goodbye_world);
