#include <linux/init.h>
#include <linux/module.h>
#include <linux/ioctl.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/err.h>
#include <linux/device.h>

#define HELLO _IO('a', 'a')

dev_t dev = 0;
static struct cdev etx_cdev;
static struct class *dev_class;

long etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    switch(cmd) {
        case HELLO:
            printk(KERN_INFO "Hello ioctl!\n");
            break;
        default:
            printk(KERN_INFO "Invalid command!\n");
            break;
    }

    return 0;
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = etx_ioctl
};

static int __init etx_driver_init(void) {
    printk(KERN_INFO "Inserting driver...\n");

    // allocate major number
    if ((alloc_chrdev_region(&dev, 0, 1, "etx_dev")) < 0) {
        printk(KERN_INFO "Cannot allocate major number!\n");
        return -1;
    }
    printk(KERN_INFO "Major = %d Minor = %d\n", MAJOR(dev), MINOR(dev));

    // create cdev structure
    cdev_init(&etx_cdev, &fops);

    // add character device to the system
    if ((cdev_add(&etx_cdev, dev, 1)) < 0) {
        printk(KERN_INFO "Cannot add the device to the system!\n");
        goto r_class;
    }
    printk(KERN_INFO "Device added to the system.\n");

    // create struct class
    if (IS_ERR(dev_class = class_create(THIS_MODULE, "etx_class"))) {
        printk(KERN_INFO "Cannot create the struct class!\n");
        goto r_class;
    }
    printk(KERN_INFO "Struct class created successfully.\n");

    // create device
    if (IS_ERR(device_create(dev_class, NULL, dev, NULL, "chrdev"))) {
        printk(KERN_INFO "Cannot create the device!\n");
        goto r_device;
    }

    printk(KERN_INFO "Device driver inserted successfully.\n");

r_device:
    class_destroy(dev_class);

r_class:
    unregister_chrdev_region(dev, 1);
    return -1;
}

static void __exit etx_driver_exit(void) {
    device_destroy(dev_class, dev);
    class_destroy(dev_class);
    cdev_del(&etx_cdev);
    unregister_chrdev_region(dev, 1);
    printk(KERN_INFO "Driver removed successfully.\n");
}

module_init(etx_driver_init);
module_exit(etx_driver_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Ocelot");
MODULE_DESCRIPTION("GA");
MODULE_VERSION("0.1");