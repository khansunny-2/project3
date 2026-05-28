CC     := gcc
CFLAGS := -Wall -O0 -fopenmp -g -mavx2 -mfma

SRCS := $(wildcard *.c)
OBJS := $(SRCS:%.c=%.o)

.PHONY: all clean

all:: $(OBJS)

$(OBJS): %.o: %.c nbody.h
	$(CC) $(CFLAGS) -c $< -o $@

clean::
	rm -f *.o
