# >--------------------------------------------------
class Test:
    def __init__(self) -> None:
        self.testing = 'class variable !'

    def decorator_factory(*args):
        print("msg from factory: ", *args)
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                print(self.testing)
                func(self, *args, **kwargs)

            return wrapper
        return decorator


    @decorator_factory('hello ?')
    def test(self, msg):
        print(msg)

# test = Test()
# test.test('this is a test')

# >--------------------------------------------------
def test(a, *args):
    print(a)
    print(args)
    print(all(args))

test('hello')
