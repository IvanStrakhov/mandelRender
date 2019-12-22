from __future__ import print_function, unicode_literals

import mandel


def main():
    print('mandel version is {}'.format(mandel.version))
    man = mandel.mandelRender()
    man.render()


if __name__ == '__main__':
    main()
