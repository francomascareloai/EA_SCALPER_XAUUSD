# MQL5 Book - Part 2 (Pages 201-400)

## Page 201

Part 3. Object Oriented Programming
201 
3.2 Classes and interfaces
class Rectangle : public Shape
{
   ...
public:
   Rectangle(int px, int py, int sx, int sy, color back) :
      Shape(px, py, back, typename(this)), dx(sx), dy(sy)
   {
      Print(__FUNCSIG__, " ", &this);
   }
};
Now we can improve the method toString using the type field.
class Shape
{
   ...
public:
   string toString() const
   {
      return type + " " + (string)coordinates.x + " " + (string)coordinates.y;
   }
};
Let's make sure that our little class hierarchy spawns objects as intended and prints test log entries
when constructors and destructors are called.
void OnStart()
{
   Shape s;
   //setting up an object by chaining calls via 'this'
   s.setColor(clrWhite).moveX(80).moveY(-50);
   Rectangle r(100, 200, 75, 50, clrBlue);
   Ellipse e(200, 300, 100, 150, clrRed);
   Print(s.toString());
   Print(r.toString());
   Print(e.toString());
}
As a result, we get approximately the following log entries (blank lines are added intentionally to
separate the output from different objects):

---

## Page 202

Part 3. Object Oriented Programming
202
3.2 Classes and interfaces
Pair::Pair(int,int) 0 0
Shape::Shape() 1048576
   
Pair::Pair(int,int) 100 200
Shape::Shape(int,int,color,string) 2097152
Rectangle::Rectangle(int,int,int,int,color) 2097152
   
Pair::Pair(int,int) 200 300
Shape::Shape(int,int,color,string) 3145728
Ellipse::Ellipse(int,int,int,int,color) 3145728
   
Shape 80 -50
Rectangle 100 200
Ellipse 200 300
   
Ellipse::~Ellipse() 3145728
Shape::~Shape() 3145728
Pair::~Pair() 200 300
   
Rectangle::~Rectangle() 2097152
Shape::~Shape() 2097152
Pair::~Pair() 100 200
   
Shape::~Shape() 1048576
Pair::~Pair() 80 -50
The log makes it clear in what order the constructors and destructors are called.
For each object, firstly, the object fields described in it are created (if there are any), and then the
base constructor and all constructors of derived classes along the inheritance chain are called. If there
are own (added) fields of some object types in a derived class, the constructors for them will be called
immediately before the constructor of this derived class. When there are several object fields, they are
created in the order in which they are described in the class.
Destructors are called in exactly the reverse order.
In the derived classes copy constructors can be defined, which we learned about in Constructors:
Default, Parametric, Copy. For specific shape types, such as a rectangle, their syntax is similar:
class Rectangle : public Shape
{
   ...
   Rectangle(const Rectangle &other) :
      Shape(other), dx(other.dx), dy(other.dy)
   {
   }
   ...
};
The scope is slightly expanding. A derived class object can be used to copy to a base class (because
the derived class contains all the data for the base class). However, in this case, of course, the fields
added in the derived class are ignored.

---

## Page 203

Part 3. Object Oriented Programming
203
3.2 Classes and interfaces
void OnStart()
{
   Rectangle r(100, 200, 75, 50, clrBlue);
   Shape s2(r);         // ok: copy derived to base
   
   Shape s;
   Rectangle r4(s);     // error: no one of the overloads can be applied 
                        // requires explicit constructor overloading
}
To copy in the opposite direction, you need to provide a constructor version with a reference to the
derived class in the base class (which, in theory, contradicts the principles of OOP), otherwise the
compilation error "no one of the overloads can be applied to the function call" will occur.
Now we can script a couple or more shape variables to then "ask" them to draw themselves using the
method draw.
void OnStart()
{
   Rectangle r(100, 200, 50, 75, clrBlue);
   Ellispe e(100, 200, 50, 75, clrGreen);
   r.draw();
   e.draw();
};
However, such an entry means that the number of shapes, their types, and parameters are hardwired
into the program, while the should be able to choose what and where to draw. Hence the need to
create shapes in a dynamic way.
3.2.1 2 Dynamic creation of objects: new and delete
So far we have only tried to create automatic objects, i.e. local variables inside OnStart. An object
declared in the global context (outside OnStart or some other function) would also be automatically
created (when the script is loaded) and deleted (when the script is unloaded).
In addition to these two modes, we have touched on the ability to describe a field of an object type (in
our example, this is the structure Pair used for the field coordinates inside the object Shape). All such
objects are also automatic: they are created for us by a compiler in a constructor of a "host" object
and deleted in its destructor.
However, it is quite often impossible to get by with only automatic objects in programs. In the case of a
drawing program, we will need to create shapes at the user's request. Moreover, shapes will need to be
stored in an array, and for this automatic objects would have to have a default constructor (which is
not the case in our case.  
For such situations, MQL5 offers the opportunity to dynamically create and delete objects. Creation is
implemented with the operator new and deletion with the operator delete.

---

## Page 204

Part 3. Object Oriented Programming
204
3.2 Classes and interfaces
Operator new
The keyword new is followed by the name of the required class and, in parentheses, a list of arguments
to call any of the existing constructors. Execution of the operator new leads to the creation of an
instance of the class.
The operator new returns a value of a special type – a pointer to an object. To describe a variable of
this type, add an asterisk character '*' after the class name. For example:
Rectangle *pr = new Rectangle(100, 200, 50, 75, clrBlue);
Here the variable pr has a type of pointer to an object of the class Rectangle. Pointers will be discussed
in more detail in a separate section.
It is important to note that the declaration of a variable of an object pointer type itself does not
allocate memory for an object and does not call its constructor. Of course, a pointer takes up space - 8
bytes, but in fact, it is an unsigned integer ulong, which the system interprets in a special way.
You can work with a pointer in the same way as with an object, i.e., you can call available methods
through the dereference operator and access fields.
Print(pr.toString());
A pointer variable that has not yet been assigned a dynamic object descriptor (for example, if the
operator new is called not at the time of initialization of a new variable, but is moved to some later lines
of the source code), contains a special null pointer, which is denoted as NULL (to distinguish it from
numbers) but is actually equal to 0.
Operator delete
Pointers received via new should be freed at the end of an algorithm using the operator delete. For
example:
delete pr;
If this is not done, the instance allocated by the operator new will remain in memory. If more and more
new objects are created in this way, and then not deleted when they are no longer needed, this will lead
to unnecessary memory consumption. The remaining unreleased dynamic objects cause warnings to be
printed when the program terminates. For example, if you don't delete the pointer pr, you'll get
something like this in the log after the script is unloaded: <segment 0809>
1 undeleted object left
1 object of type Rectangle left
168 bytes of leaked memory
The terminal reports how many objects and what class were forgotten by the programmer, as well as
how much memory they occupied.
Once the operator delete is called for a pointer, the pointer is invalidated because the object no longer
exists. A subsequent attempt to access its properties causes a run-time error "Invalid pointer
accessed":

---

## Page 205

Part 3. Object Oriented Programming
205
3.2 Classes and interfaces
Critical error while running script 'shapes (EURUSD,H1)'.
Invalid pointer access.
The MQL program is then interrupted.
This, however, does not mean that the same pointer variable can no longer be used. It is enough to
assign a pointer to another newly created instance of the object.
MQL5 has a built-in function that allows you to check the validity of a pointer in a variable –
CheckPointer:
ENUM_POINTER_TYPE CheckPointer(object *pointer);
It takes one parameter of a pointer to a type class and returns a value from the ENUM_POINTER_TYPE
enumeration:
• POINTER_INVALID – incorrect pointer;
• POINTER_DYNAMIC – valid pointer to a dynamic object;
• POINTER_AUTOMATIC – valid pointer to an automatic object.
Execution of the statement delete only makes sense for a pointer for which the function returned
POINTER_DYNAMIC. For an automatic object, it will have no effect (such objects are deleted
automatically when control returns from the block of code in which the variable is defined).
The following macro simplifies and ensures the correct cleanup for a pointer:
#define FREE(P) if(CheckPointer(P) == POINTER_DYNAMIC) delete (P)
The necessity to explicitly "clean up" is an inevitable price to pay for the flexibility provided by dynamic
objects and pointers.
3.2.1 3 Pointers
As we said in the Class Definition section, pointers in MQL5 are some descriptors (unique numbers) of
objects, and not addresses in memory, as in C++. For an automatic object, we obtained a pointer by
putting an ampersand in front of its name (in this context, the ampersand character is the "get
address" operator). So, in the following example, the variable p points to the automatic object s.
Shape s;        // automatic object
Shape *p = &s;  // a pointer to the same object
s.draw();       // calling an object method
p.draw();       // doing the same
In the previous sections, we learned how to get a pointer to an object as a result of creating it
dynamically with new. At this time, an ampersand is not needed to get a descriptor: the value of the
pointer is the descriptor.
The MQL5 API provides the function GetPointer which performs the same action as the ampersand
operator '&', i.e. returns a pointer to an object:
void *GetPointer(Class object);
Which of the two options to use is a matter of preference.

---

## Page 206

Part 3. Object Oriented Programming
206
3.2 Classes and interfaces
Pointers are often used to link objects together . Let's illustrate the idea of   creating subordinate objects
that receive a pointer to this of its object-creator (ThisCallback.mq5). We mentioned this trick in the
section on the keyword this.
Let's try using it to implement a scheme for notifying the "creator" from time to time about the
percentage of calculations performed in the subordinate object: we made its analog using the function
pointer. The class Manager controls calculations, and the calculations themselves (most probably, using
different formulas) are performed in separate classes - in this example, one of them, the class Element
is shown.

---

## Page 207

Part 3. Object Oriented Programming
207
3.2 Classes and interfaces
class Manager; // preliminary announcement
   
class Element
{
   Manager *owner; // pointer
   
public:
   Element(Manager &t): owner(&t) { }
   
   void doMath()
   {
      const int N = 1000000;
      for(int i = 0; i < N; ++i)
      {
         if(i % (N / 20) == 0)
         {
            // we pass ourselves to the method of the control class
            owner.progressNotify(&this, i * 100.0f / N);
         }
         // ... massive calculations
      }
   }
   
   string getMyName() const
   {
      return typename(this);
   }
};
   
class Manager
{
   Element *elements[1]; // array of pointers (1 for demo)
   
public:
   Element *addElement()
   {
      // looking for an empty slot in the array
      // ...
      // passing to the constructor of the subclass
      elements[0] = new Element(this); // dynamic creation of an object
      return elements[0];
   }
   
   void progressNotify(Element *e, const float percent)
   {
      // Manager chooses how to notify the user:
      // display, print, send to the Internet
      Print(e.getMyName(), "=", percent);
   }
};

---

## Page 208

Part 3. Object Oriented Programming
208
3.2 Classes and interfaces
A subordinate object can use the received link to notify the "boss" about the work progress. Reaching
the end of the calculation sends a signal to the control object that it is possible to delete the calculator
object, or let another one work. Of course, the fixed one-element array in the class Manager doesn't
look very impressive, but as a demonstration, it gets the point across. The manager not only manages
the distribution of computing tasks, but also provides an abstract layer for notifying the user: instead of
outputting to a log, it can write messages to a separate file, display them on the screen, or send them
to the Internet.
By the way, pay attention to the preliminary declaration of the class Manager before the class
definition Element. It is needed to describe in the class Element a pointer to the class Manager, which
is defined below in the code. If the forward declaration is omitted, we get the error "'Manager' -
unexpected token, probably type is missing?".
The need for forward declaration arises when two classes refer to each other through their members: in
this case, in whatever order we arrange the classes, it is impossible to fully define either of them. A
forward declaration allows you to reserve a type name without a full definition.
A fundamental property of pointers is that a pointer to a base class can be used to point to an
object of any derived class. This is one of the manifestations of polymorphism. This behavior is
possible because derived objects contain built-in "sub-objects" of parent classes like nesting dolls
matryoshkas.
In particular, for our task with shapes, it is easy to describe a dynamic array of pointers Shape and add
objects of different types to it at the request of the user.
The number of classes will be expanded to five (Shapes2.mq5). In addition to Rectangle and Ellipse, let's
add Triangle, and also make a class derived from Rectangle for a square (Square), and a class derived
from Ellipse for a circle (Circle). Obviously, a square is a rectangle with equal sides, and a circle is an
ellipse with the equal large and small radii.
To pass a string class name along the inheritance chain, let's add in the protected sections of the
classes Rectangle and Ellipse special constructors with an additional string parameter t:
class Rectangle : public Shape
{
protected:
   Rectangle(int px, int py, int sx, int sy, color back, string t) :
      Shape(px, py, back, t), dx(sx), dy(sy)
   {
   }
   ...
};
Then, when creating a square, we set not only equal sizes of the sides but also pass typename(this)
from the class Square:

---

## Page 209

Part 3. Object Oriented Programming
209
3.2 Classes and interfaces
class Square : public Rectangle
{
public:
   Square(int px, int py, int sx, color back) :
      Rectangle(px, py, sx, sx, back, typename(this))
   {
   }
};
In addition, we will move constructors in the class Shape to the protected section: this will prohibit the
creation of the object Shape by itself - it can only act as a base for their descendant classes.
Let's assign the function addRandomShape to generate shapes, which returns a pointer to a newly
created object. For demonstration purposes, it will now implement a random generation of shapes: their
types, positions, sizes and colors.
Supported shape types are summarized in the SHAPES enumeration: they correspond to five
implemented classes.
Random numbers in a given range are returned by the function random (it uses the built-in function
rand, which returns a random integer in the range from 0 to 32767 each time it is called. The centers
of the shapes are generated in the range from 0 to 500 pixels, the sizes of the shapes are in the range
of up to 200. The color is formed from three RGB components (see Color section), each ranging from 0
to 255.

---

## Page 210

Part 3. Object Oriented Programming
21 0
3.2 Classes and interfaces
int random(int range)
{
   return (int)(rand() / 32767.0 * range);
}
   
Shape *addRandomShape()
{
   enum SHAPES
   {
      RECTANGLE,
      ELLIPSE,
      TRIANGLE,
      SQUARE,
      CIRCLE,
      NUMBER_OF_SHAPES
   };
   
   SHAPES type = (SHAPES)random(NUMBER_OF_SHAPES);
   int cx = random(500), cy = random(500), dx = random(200), dy = random(200);
   color clr = (color)((random(256) << 16) | (random(256) << 8) | random(256));
   switch(type)
   {
      case RECTANGLE:
         return new Rectangle(cx, cy, dx, dy, clr);
      case ELLIPSE:
         return new Ellipse(cx, cy, dx, dy, clr);
      case TRIANGLE:
         return new Triangle(cx, cy, dx, clr);
      case SQUARE:
         return new Square(cx, cy, dx, clr);
      case CIRCLE:
         return new Circle(cx, cy, dx, clr);
   }
   return NULL;
}
   
void OnStart()
{
   Shape *shapes[];
   
   // simulate the creation of arbitrary shapes by the user
   ArrayResize(shapes, 10);
   for(int i = 0; i < 10; ++i)
   {
      shapes[i] = addRandomShape();
   }
   
   // processing shapes: for now, just output to the log 
   for(int i = 0; i < 10; ++i)
   {
      Print(i, ": ", shapes[i].toString());

---

## Page 211

Part 3. Object Oriented Programming
21 1 
3.2 Classes and interfaces
      delete shapes[i];
   }
}
We generate 1 0 shapes and output them to the log (the result may differ due to the randomness of the
choice of types and properties). Don't forget to delete the objects with delete because they were
created dynamically (here this is done in the same loop because the shapes are not used further; in a
real program, the array of shapes will most likely be stored somehow to a file for later loading and
continuing to work with an image).
0: Ellipse 241 38
1: Rectangle 10 420
2: Circle 186 38
3: Triangle 27 225
4: Circle 271 193
5: Circle 293 57
6: Rectangle 71 424
7: Square 477 46
8: Square 366 27
9: Ellipse 489 105
The shapes are successfully created and inform about their properties.
We are now ready to access the API of our classes, i.e. the draw method.
3.2.1 4 Virtual methods (virtual and override)
Classes are intended to describe external programming interfaces and provide their internal
implementation. Since the functionality of our test program is to draw various shapes, we have
described several variables in the class Shape and its descendants for future implementation, and also
reserved the method draw for the interface.
In the base class Shape, it shouldn't and can't do anything because Shape is not a concrete shape:
we'll convert Shape to an abstract class later (we will talk more about abstract classes and interfaces
later).
Let's redefine the draw method in the Rectangle, Ellipse and other derived classes (Shapes3.mq5). This
involves copying the method and modifying its content accordingly. Although many refer to this process
as "overriding", we will distinguish between the two terms, reserving "overriding" exclusively for virtual
methods, which will be discussed later.
Strictly speaking, redefining a method only requires the method name to match. However, to ensure
consistent usage throughout the code, it is essential to maintain the same parameter list and return
type.

---

## Page 212

Part 3. Object Oriented Programming
21 2
3.2 Classes and interfaces
class Rectangle : public Shape
{
   ...
   void draw()
   {
      Print("Drawing rectangle");
   }
};
Since we don't know how to draw on the screen yet, we'll just output the message to the log.
It is important to note that by providing a new implementation of the method in the derived class, we
thereby get 2 versions of the method: one refers to the built-in base object (inner Shape), and the
other to the derived one (outer Rectangle).
The first will be called for a variable of type Shape, and the second one for a variable of type Rectangle.
In a longer inheritance chain, a method can be redefined and propagated even more times.
You can change an access type of a new method, for example, make it public if it was protected, or
vice versa. But in this case, we left the draw method in the public section.
If necessary, the programmer can call the implementation of the method of any of the progenitor
classes: for this, a special context resolution operator is used — two colons '::'. In particular, we could
call the draw implementation from the class Rectangle from the method draw of the class Square: for
this, we specify the name of the desired class, '::' and the method name, for example,
Rectangle::draw(). Calling draw without specifying the context implies a method of the current class,
and therefore if you do it from the method draw itself, you will get an infinite recursion, and ultimately,
a stack overflow and program crash.
class Square : public Rectangle
{
public:
   ...
   void draw()
   {
      Rectangle::draw();
      Print("Drawing square");
   }
};
Then calling draw on the object Square would log two lines:
   Square s(100, 200, 50, clrGreen);
   s.draw(); // Drawing rectangle
             // Drawing square
Binding a method to a class in which it is declared provides the static dispatch (or static binding): the
compiler decides which method to call at the compilation stage and "hardwires" the found match into
binary code.
During the decision process, the compiler looks for the method to be called in the object of the class for
which the dereference ('.') is performed. If the method is present, it is called, and if not, the compiler
checks the parent class for the presence of the method, and so on, through the inheritance chain until

---

## Page 213

Part 3. Object Oriented Programming
21 3
3.2 Classes and interfaces
the method is found. If the method is not found in any of the classes in the chain, an "undeclared
identifier" compilation error will occur.
In particular, the following code calls the setColor method on the object Rectangle:
   Rectangle r(100, 200, 75, 50, clrBlue);
   r.setColor(clrWhite);
However, this method is defined only in the base class Shape and is built in once in all descendant
classes, and therefore it will be executed here.
Let's try to start drawing arbitrary shapes from an array in the function OnStart (recall that we have
duplicated and modified the method draw in all descendant classes).
   for(int i = 0; i < 10; ++i)
   {
      shapes[i].draw();
   }
Oddly enough, nothing is output to the log. This happens because the program calls the method draw of
the class Shape.
There is a major drawback of static dispatch here: when we use a pointer to a base class to store an
object of a derived class, the compiler chooses a method based on the type of the pointer, not the
object. The fact is that at the compilation stage, it is not yet known what class object it will point to
during program execution.
Thus, there is a need for a more flexible approach: a dynamic dispatch (or binding), which would defer
the choice of a method (from among all the overridden versions of the method in the descendant chain)
to runtime. The choice must be made based on analysis of the actual class of the object at the pointer.
It is dynamic dispatching that provides the principle of polymorphism.
This approach is implemented in MQL5 using virtual methods. In the description of such a method, the
keyword virtual must be added at the beginning of the header.
Let's declare the method draw in the class Shape (Shapes4.mq5) as virtual. This will automatically
make all versions of it in derived classes virtual as well.
class Shape
{
   ...
   virtual void draw()
   {
   }
};
Once a method is virtualized, modifying it in derived classes is called overriding rather than redefinition.
Overriding requires the name, parameter types, and return value of the method to match (taking into
account the presence/absence of const modifiers).
Note that overriding virtual functions is different from function overloading. Overloading uses the
same function name, but with different parameters (in particular, we saw the possibility of
overloading a constructor in the example of structures, see Constructors and Destructors), and
overriding requires full matching of function signatures.
Overridden functions must be defined in different classes that are related by inheritance

---

## Page 214

Part 3. Object Oriented Programming
21 4
3.2 Classes and interfaces
relationships. Overloaded functions must be in the same class — otherwise, it will not be an
overload but, most likely, a redefinition (and it will work differently, see further analysis of the
example OverrideVsOverload.mq5).
If you run a new script, the expected lines will appear in the log, signaling calls to specific versions of
the draw method in each of the classes.
Drawing square
Drawing circle
Drawing triangle
Drawing ellipse
Drawing triangle
Drawing rectangle
Drawing square
Drawing triangle
Drawing square
Drawing triangle
In derived classes where a virtual method is overridden, it is recommended to add the keyword override
to its header (although this is not required).
class Rectangle : public Shape
{
   ...
   void draw() override
   {
      Print("Drawing rectangle");
   }
};
This allows the compiler to know that we are overriding the method on purpose. If in the future the API
of the base class suddenly changes and the overridden method is no longer virtual (or simply removed),
the compiler will generate an error message: "method is declared with 'override' specifier, but does not
override any base class method". Keep in mind that even adding or removing the modifier const from a
method changes its signature, and overriding may be broken due to this.
The keyword virtual before an overridden method is also allowed but not required.
For dynamic dispatching to work, the compiler generates a table of virtual functions for each class. An
implicit field is added to each object with a link to the given table of its class. The table is populated by
the compiler based on information about all virtual methods and their overridden versions along the
inheritance chain of a particular class.
A call to a virtual method is encoded in the binary image of the program in a special way: first, the
table is looked up in search of a version for a class of a particular object (located at the pointer), and
then a transition is made to the appropriate function.
As a result, dynamic dispatch is slower than static dispatch.
In MQL5, classes always contain a table of virtual functions, regardless of the presence of virtual
methods.
If a virtual method returns a pointer to a class, then when it is overridden, it is possible to change
(make it more specific, highly specialized) the object type of the return value. In other words, the type

---

## Page 215

Part 3. Object Oriented Programming
21 5
3.2 Classes and interfaces
of the pointer can be not only the same as in the initial declaration of the virtual method but also any of
its successors. Such types are called "covariant" or interchangeable.
For example, if we made the method setColor virtual in the class Shape:
class Shape
{
   ...
   virtual Shape *setColor(const color c)
   {
      backgroundColor = c;
      return &this;
   }
   ...
};
we could override it in the class Rectangle like this (only as a demonstration of the technology):
class Rectangle : public Shape
{
   ...
   virtual Rectangle *setColor(const color c) override
   {
      // call original method
      // (by pre-lightening the color,
      // no matter what for)
      Rectangle::setColor(c | 0x808080);
      return &this;
   }
};
Note that the return type is a pointer to Rectangle instead of Shape.
It makes sense to use a similar trick if the overridden version of the method changes something in that
part of the object that does not belong to the base class, so that the object, in fact, no longer
corresponds to the allowed state (invariant) of the base class.
Our example with drawing shapes is almost ready. It remains to fill the virtual methods draw with real
content. We will do this in the chapter Graphics (see example Obj ectShapesDraw.mq5), but we will
improve it after studying graphic resources.
Taking into account the inheritance concept, the procedure by which the compiler chooses the
appropriate method looks a bit confusing. Based on the method name and the specific list of
arguments (their types) in the call instruction, a list of all available candidate methods is compiled.
For non-virtual methods, at the beginning only methods of the current class are analyzed. If none of
them matches, the compiler will continue searching the base class (and then more distant
ancestors until it finds a match). If among the methods of the current class, there is a suitable one
(even if the implicit conversion of argument types is necessary), it will be picked. If the base class
had a method with more appropriate argument types (no conversion or fewer conversions), the
compiler still won't get to it. In other words, non-virtual methods are analyzed starting from the
class of the current object towards the ancestors to the first "working" match.
For virtual methods, the compiler first finds the required method by name in the pointer class and

---

## Page 216

Part 3. Object Oriented Programming
21 6
3.2 Classes and interfaces
then selects the implementation in the table of virtual functions for the most instantiated class
(furthest descendant) in which this method is overridden in the chain between the pointer type and
the object type. In this case, implicit argument conversion can also be used if there is no exact
match between the types of arguments.
Let's consider the following example (OverrideVsOverload.mq5). There are 4 classes that are chained:
Base, Derived, Concrete and Special. All of them contain methods with type arguments int and float. In
the function OnStart, the integer i and the real f variables are used as arguments for all method calls.

---

## Page 217

Part 3. Object Oriented Programming
21 7
3.2 Classes and interfaces
class Base
{
public:
   void nonvirtual(float v)
   {
      Print(__FUNCSIG__, " ", v);
   }
   virtual void process(float v)
   {
      Print(__FUNCSIG__, " ", v);
   }
};
class Derived : public Base
{
public:
   void nonvirtual(int v)
   {
      Print(__FUNCSIG__, " ", v);
   }
   virtual void process(int v) // override
   // error: 'Derived::process' method is declared with 'override' specifier,
   // but does not override any base class method
   {
      Print(__FUNCSIG__, " ", v);
   }
};
class Concrete : public Derived
{
};
class Special : public Concrete
{
public:
   virtual void process(int v) override
   {
      Print(__FUNCSIG__, " ", v);
   }
   virtual void process(float v) override
   {
      Print(__FUNCSIG__, " ", v);
   }
};
First, we create an object of class Concrete and a pointer to it Base *ptr. Then we call non-virtual and
virtual methods for them. In the second part, the methods of the object Special are called through the
class pointers Base and Derived.

---

## Page 218

Part 3. Object Oriented Programming
21 8
3.2 Classes and interfaces
void OnStart()
{
   float f = 2.0;
   int i = 1;
   Concrete c;
   Base *ptr = &c;
   
   // Static link tests
   ptr.nonvirtual(i); // Base::nonvirtual(float), conversion int -> float
   c.nonvirtual(i);   // Derived::nonvirtual(int)
   // warning: deprecated behavior, hidden method calling
   c.nonvirtual(f);   // Base::nonvirtual(float), because
                      // method selection ended in Base,
                      // Derived::nonvirtual(int) does not suit to f
   // Dynamic link tests
   // attention: there is no method Base::process(int), also
   // there are no process(float) overrides in classes up to and including Concrete
   ptr.process(i);    // Base::process(float), conversion int -> float
   c.process(i);      // Derived::process(int), because
                      // there is no override in Concrete,
                      // and the override in Special does not count
   Special s;
   ptr = &s;
   // attention: there is no method Base::process(int) in ptr
   ptr.process(i);    // Special::process(float), conversion int -> float
   ptr.process(f);    // Special::process(float)
   Derived *d = &s;
   d.process(i);      // Special::process(int)
   // warning: deprecated behavior, hidden method calling
   d.process(f);      // Special::process(float)
}
The log output is shown below.

---

## Page 219

Part 3. Object Oriented Programming
21 9
3.2 Classes and interfaces
void Base::nonvirtual(float) 1.0
void Derived::nonvirtual(int) 1
void Base::nonvirtual(float) 2.0
void Base::process(float) 1.0
void Derived::process(int) 1
void Special::process(float) 1.0
void Special::process(float) 2.0
void Special::process(int) 1
void Special::process(float) 2.0
The ptr.nonvirtual(i) call is made using static binding, and the integer i is preliminarily cast to the
parameter type, float.
The call c.nonvirtual(i) is also static, and since there is no void nonvirtual(int) method in the class
Concrete, the compiler finds such a method in the parent class Derived.
Calling the function of the same name on the same object with a value of type float leads the compiler
to the method Base::nonvirtual(float) because Derived::nonvirtual(int) is not suitable (the conversion
would lead to a loss of precision). Along the way, the compiler issues a "deprecated behavior, hidden
method calling" warning.
Overloaded methods may look like overridden (have the same name but different parameters) but they
are different because they are located in different classes. When a method in a derived class overrides
a method in a parent class, it replaces the behavior of the parent class method which can sometimes
cause unexpected effects. The programmer might expect the compiler to choose another suitable
method (as in overloading), but instead the subclass is invoked.
To avoid potential warnings, if the implementation of the parent class is necessary, it should be written
as exactly the same function in the derived class, and the base class should be called from it.
class Derived : public Base
{
public:
   ...
   // this override will suppress the warning
   // "deprecated behavior, hidden method calling"
   void nonvirtual(float v)
   {
      Base::nonvirtual(v);
      Print(__FUNCSIG__, " ", v);
   }
...
Let's go back to tests in OnStart.
Calling ptr.process(i) demonstrates the confusion between overloading and overriding described above.
The Base class has a process(float) virtual method, and the class Derived adds a new virtual method
process(int), which is not overriding in this case because parameter types are different. The compiler
selects a method by name in the base class and checks the virtual function table for overrides in the
inheritance chain up to the Concrete class (inclusive, this is the object class by pointer). Since no
overrides were found, the compiler took Base::process(float) and applied the type conversion of the
argument to the parameter (int to float).

---

## Page 220

Part 3. Object Oriented Programming
220
3.2 Classes and interfaces
If we followed the rule of always writing the word override where redefining is implied and added it to
Derived, we would get an error:
class Derived : public Base
{
   ...
   virtual void process(int v) override // error!
   {
      Print(__FUNCSIG__, " ", v);
   }
};
The compiler would report "'Derived::process' method is declared with 'override' specifier, but does not
override any base class method". This would serve as a hint to fixing the problem.
Calling process(i) on the Concrete object is done with Derived::process(int). Although we have an even
further redefinition in the class Special, it is irrelevant because it's done in the inheritance chain after
the Concrete class.
When the pointer ptr is later assigned to the Special object, calls to process(i) and process(f) are
resolved by the compiler as Special::process(float) because Special overrides Base::process(float). The
choice of the float parameter occurs for the same reason as described earlier: the method
Base::process(float) is overridden by Special.
If we apply the pointer d of type Derived, then we finally get the expected call Special::process(int) for
the string d.process(i). The point is that process(int) is defined in Derived, and falls into the scope of the
compiler's search.
Note that the Special class not only overrides the inherited virtual methods but also overloads two
methods in the class itself.
Do not call a virtual function from a constructor or destructor! While technically possible, the virtual
behavior in the constructor and destructor is completely lost and you might get unexpected results.
Not only explicit but also indirect calls should be avoided (for example, when a simple method is
called from a constructor, which in turn calls a virtual one).
Let's analyze the situation in more detail using the example of a constructor. The fact is that at the
time of the constructor's work, the object is not yet fully assembled along the entire inheritance
chain, but only up to the current class. All derived part have yet to be "finished" around the existing
core. Therefore, all later virtual method overrides (if any) are not yet available at this point. As a
result, the current version of the method will be called from the constructor.
3.2.1 5 Static members
So far, we have considered the fields and methods of a class that describe the state and behavior of
objects of a given class. However, in programs, it may be necessary to store certain attributes or
perform operations on the entire class, rather than on its objects. Such class properties are called
static and are described using the static keyword added before the type. They are also supported in
structures and unions.
For example, we can count the number of shapes created by the user in a drawing program. To do this,
in the class Shape, we will describe the static variable count(Shapes5.mq5).

---

## Page 221

Part 3. Object Oriented Programming
221 
3.2 Classes and interfaces
class Shape
{
private:
   static int count;
   
protected:
   ...
   Shape(int px, int py, color back, string t) :
      coordinates(px, py),
      backgroundColor(back),
      type(t)
   {
      ++count;
   }
   
public:
   ...
   static int getCount()
   {
      return count;
   }
};
It is defined in the private section and therefore not accessible from the outside.
To read the current counter value, a public static method getCount() is provided. In theory, since static
members are defined in the context of a class, they receive visibility restrictions according to the
modifier of the section in which they are located.
We will increase the counter by 1  in the parametric constructor Shape, and remove the default
constructor. Thus, each instance of a shape of any derived type will be taken into account.
Note that a static variable must be explicitly defined (and optionally initialized) outside the class block:
static int Shape::count = 0;
Static class variables are similar to global variables and static variables inside functions (see section
Static variables) in the sense that they are created when the program starts and are deleted before it
is unloaded. Therefore, unlike object variables, they must exist from the beginning as a single instance.
In this case, zero-initialization can be omitted because, as we know, global and static variables are set
to zero by default. Arrays can also be static.
In the definition of a static variable, we see the use of the special context selection operator '::'. With
it, a fully qualified variable name is formed. To the left of '::' is the name of the class to which the
variable belongs, and to the right is its identifier. Obviously, the fully qualified name is necessary,
because within different classes static variables with the same identifier can be declared, and a way to
uniquely refer to each of them is needed.
The same '::' operator is used to access not only public static class variables but also methods. In
particular, in order to call the method getCount in the OnStart function, we use the syntax
Shape::getCount():

---

## Page 222

Part 3. Object Oriented Programming
222
3.2 Classes and interfaces
void OnStart()
{
   for(int i = 0; i < 10; ++i)
   {
      Shape *shape = addRandomShape();
      shape.draw();
      delete shape;
   }
   
   Print(Shape::getCount()); // 10
}
Since the specified number of shapes (1 0) is now being generated, we can verify that the counter is
working correctly.
If you have a class object, you can refer to a static method or property through the usual dereference
(for example, shape.getCount()), but such a notation can be misleading (because it hides the fact that
the object is actually not accessed).
Note that the creation of derived classes does not affect static variables and methods in any way: they
are always assigned to the class in which they were defined. Our counter is the same for all classes of
shapes derived from Shape.
You can't use this inside static methods because they are executed without being tied to a specific
object. Also, from a static method, you cannot directly, without dereferencing any object type variable,
call a regular class method or access its field. For example, if you call draw from getCount, you get an
"access to non-static member or function" error:
   static int getCount()
   {
      draw(); // error: 'draw' - access to non-static member or function
      return count;
   }
For the same reason, static methods cannot be virtual.
Is it possible, using static variables, to calculate not the total number of shapes, but their statistics by
type? Yes, it is possible. This task is left for independent study. Those interested can find one of the
implementation examples in the script Shapes5stats.mq5.
3.2.1 6 Nested types, namespaces, and the context operator '::'
Classes, structures, and unions can be described not only in the global context but also within another
class or structure. And even more: the definition can be done inside the function. This allows you to
describe all the entities necessary for the operation of any class or structure within the appropriate
context and thereby avoid potential name conflicts.
In particular, in the drawing program, the structure for storing coordinates Pair has been defined
globally so far. As the program grows, it is quite possible that another entity called Pair will be needed
(especially given the rather generic name). Therefore, it is desirable to move the description of the
structure inside the class Shape (Shapes6.mq5).

---

## Page 223

Part 3. Object Oriented Programming
223
3.2 Classes and interfaces
class Shape
{
public:
   struct Pair
   {
      int x, y;
      Pair(int a, int b): x(a), y(b) { }
   };
   ...
};
The nested descriptions have access permissions in accordance with the specified section modifiers. In
this case, we have made the name Pair publicly available. Inside the class Shape, the handling of the
Pair structure type does not change in any way due to the transfer. However, in external code, you
must specify a fully qualified name that includes the name of the external class (context), the context
selection operator '::' and the internal entity identifier itself. For example, to describe a variable with a
pair of coordinates, you would write:
Shape::Pair coordinates(0, 0);
The level of nesting when describing entities is not limited, so a fully qualified name can contain
identifiers of multiple levels (contexts) separated by '::'. For example, we could wrap all drawing classes
inside the outer class Drawing, in the public section.
class Drawing
{
public:
   class Shape
   {
   public:
      struct Pair
      {
         ...
      };
   };
   class Rectangle : public Shape
   {
      ...
   };
   ...
};
Then fully qualified type names (e.g. for use in OnStart or other external functions) would be
lengthened:
Drawing::Shape::Rect coordinates(0, 0);
Drawing::Rectangle rect(200, 100, 70, 50, clrBlue);
On the one hand, this is inconvenient, but on the other hand, it is sometimes a necessity in large
projects with a large number of classes. In our small project, this approach is used only to demonstrate
the technical feasibility.
To combine logically related classes and structures into named groups, MQL5 provides an easier way
than including them in an "empty" wrapper class.

---

## Page 224

Part 3. Object Oriented Programming
224
3.2 Classes and interfaces
A namespace is declared using the keyword namespace followed by the name and a block of curly
braces that includes all the necessary definitions. Here's what the same paint program looks like using
namespace:
namespace Drawing
{
   class Shape
   {
   public:
      struct Pair
      {
         ...
      };
   };
   class Rectangle : public Shape
   {
      ...
   };
   ...
}
There are two main differences: the internal contents of the space are always available publicly (access
modifiers are not applicable in it) and there is no semicolon after the closing curly brace.
Let's add the method move to the class Shape, which takes the structure Pair as a parameter:
class Shape
{
public:
   ...
   Shape *move(const Pair &pair)
   {
      coordinates.x += pair.x;
      coordinates.y += pair.y;
      return &this;
   }
};
Then, in the function OnStart, you can organize the shift of all shapes by a given value by calling this
function:

---

## Page 225

Part 3. Object Oriented Programming
225
3.2 Classes and interfaces
void OnStart()
{
   //draw a random set of shapes
   for(int i = 0; i < 10; ++i)
   {
      Drawing::Shape *shape = addRandomShape();
      // move all shapes
      shape.move(Drawing::Shape::Pair(100, 100));
      shape.draw();
      delete shape;
   }
}
Note that the types Shape and Pair have to be described with full names: Drawing::Shape and
Drawing::Shape::Pair respectively.
There may be several blocks with the same space name: all their contents will fall into one logically
unified context with the specified name.
Identifiers defined in the global context, in particular all built-in functions of the MQL5 API, are also
available through the context selection operator not preceded by any notation. For example, here's
what a call to the function Print might look like:
::Print("Done!");
When the call is made from any function defined in the global context, there is no need for such an
entry.
Necessity can manifest itself inside any class or structure if an element of the same name (function,
variable or constant) is defined in them. For example, let's add the method Print to the class Shape:
   static void Print(string x)
   {
      // empty
      // (likely will output it to a separate log file later)
   }
Since the test implementations of the draw method in derived classes call Print, they are now
redirected to this Print method: from several identical identifiers, the compiler chooses the one that is
defined in a closer context. In this case, the definition in the base class is closer to the shapes than the
global context. As a result, logging output from shape classes will be suppressed.
However, calling Print from the function OnStart still works (because it is outside the context of the
class Shape).
void OnStart()
{
   ...
   Print("Done!");
}
To "fix" debug printing in classes, you need to precede all Print calls with a global context selection
operator:

---

## Page 226

Part 3. Object Oriented Programming
226
3.2 Classes and interfaces
class Rectangle : public Shape
{
   ...
   void draw() override
   {
      ::Print("Drawing rectangle"); // reprint via global Print(...)
   }
};
3.2.1 7 Splitting class declaration and definition
In large software projects, it is convenient to separate classes into a brief description (declaration) and
a definition, which includes the main implementation details. In some cases, such a separation becomes
necessary if the classes somehow refer to each other, that is, none can be fully defined without prior
declarations.
We saw an example of a forward declaration in the section Indicators (see file ThisCallback.mq5), where
classes Manager and Element contain reciprocal pointers. There, the class was pre-declared in a short
form: in the form of a header with the keyword class and a name:
class Manager;
However, this is the shortest declaration possible. It registers only the name and makes it possible to
postpone the description of the programming interface until some time, but this description must be
encountered somewhere later in the code.
More often, the declaration includes a complete description of the interface: it specifies all the variables
and method headers of the class but without their bodies (code blocks).
Method definitions are written separately: with headers that use fully qualified names that include the
name of the class (or multiple classes and namespaces if the method context is highly nested). The
names of all classes and the name of the method are concatenated using the context selection
operator '::'.
type class_name [:: nested_class_name...] :: method_name([parameters...])
{
}
In theory, you can define part of the methods directly in the class description block (usually they do
this with small functions), and some can be taken out separately (as a rule, large functions). But a
method must have only one definition (that is, you cannot define a method in a class block, and then
again separately) and one declaration (a definition in a class block is also a declaration).
The list of parameters, return type and const modifiers (if any) must match exactly in the method
declaration and definition.
Let's see how we can separate the description and definition of classes from the script
ThisCallback.mq5 (an example from the section Pointers): let's create its analog with the name
ThisCallback2.mq5.
The predeclaration Manager will still come at the beginning. Further, both classes Element and Manager
are declared without implementation: instead of a block of code with a method body, there is a
semicolon.

---

## Page 227

Part 3. Object Oriented Programming
227
3.2 Classes and interfaces
class Manager; // preliminary announcement
  
class Element
{
   Manager *owner; // pointer
public:
   Element(Manager &t);
   void doMath();
   string getMyName() const;
};
  
class Manager
{
   Element *elements[1]; // array of pointers (replace with dynamic)
public:
   ~Manager();
   Element *addElement();
   void progressNotify(Element *e, const float percent);
};
The second part of the source code contains implementations of all methods (the implementations
themselves are unchanged).

---

## Page 228

Part 3. Object Oriented Programming
228
3.2 Classes and interfaces
Element::Element(Manager &t) : owner(&t)
{
}
void Element::doMath()
{
   ...
}
string Element::getMyName() const
{
   return typename(this);
}
Manager::~Manager()
{
   ...
}
Element *Manager::addElement()
{
   ...
}
void Manager::progressNotify(Element *e, const float percent)
{
   ...
}
Structures also support separate method declarations and definitions.
Note that the constructor initialization list (after the name and ':') is a part of the definition and
therefore must precede the function body (in other words, the initialization list is not allowed in a
constructor declaration where only the header is present).
Separate writing of the declaration and definition allows the development of libraries, the source code
of which must be closed. In this case, the declarations are placed in a separate header file with the
mqh extension, while the definitions are placed in a file of the same name with the mq5 extension. The
program is compiled and distributed as an ex5 file with a header file describing the external interface.
In this case, the question may arise why part of the internal implementation, in particular the
organization of data (variables), is visible in the external interface. Strictly speaking, this signals an
insufficient level of abstraction in the class hierarchy. All classes that provide an external interface
should not expose any implementation details.
In other words, if we set ourselves the goal of exporting the above classes from a certain library, then
we would need to separate their methods into base classes that would provide a description of the API
(without data fields), and Manager and Element inherit from them. At the same time, in the methods of
base classes, we cannot use any data from derived classes and, by and large, they cannot have
implementations at all. How is it possible?
To do this, there is a technology of abstract methods, abstract classes and interfaces.

---

## Page 229

Part 3. Object Oriented Programming
229
3.2 Classes and interfaces
3.2.1 8 Abstract classes and interfaces 
To explore abstract classes and interfaces, let's go back to our end-to-end drawing program example.
Its API for simplicity consists of a single virtual method draw. Until now, it has been empty, but at the
same time, even such an empty implementation is a concrete implementation. However, objects of the
class Shape cannot be drawn - their shape is not defined. Therefore, it makes sense to make the
method draw abstract or, as it is otherwise called, purely virtual.
To do this, the block with an empty implementation should be removed, and "= 0" should be added to
the method header:
class Shape
{
public:
   virtual void draw() = 0;
   ...
A class that has at least one abstract method also becomes abstract, because its object cannot be
created: there is no implementation. In particular, our constructor Shape was available to derived
classes (thanks to the protected modifier), and their developers could, hypothetically, create an object
Shape. But it was like that before, and after the declaration of the abstract method, we stopped this
behavior, as it was forbidden by us, the authors of the drawing interface. The compiler will throw an
error:
'Shape' -cannot instantiate abstract class
      'void Shape::draw()' is abstract
The best approach to describe an interface is to create an abstract class for it, containing only
abstract methods. In our case, the method draw should be moved to the new class Drawable, and the
class Shape should be inherited from it (Shapes.mq5).
class Drawable
{
public:
   virtual void draw() = 0;
};
class Shape : public Drawable
{
public:
   ...
   // virtual void draw() = 0; // moved to base class
   ...
};
Of course, interface methods must be in the section public.
MQL5 provides another convenient way to describe interfaces by using the keyword interface. All
methods in an interface are declared without implementation and are considered public and virtual. The
description of the Drawable interface which is equivalent to the above class looks like this:

---

## Page 230

Part 3. Object Oriented Programming
230
3.2 Classes and interfaces
interface Drawable
{
   void draw();
};
In this case, nothing needs to be changed in the descendant classes if there were no fields in the
abstract class (which would be a violation of the abstraction principle).
Now it's time to expand the interface and make the trio of methods setColor, moveX, moveY also part
of it.
interface Drawable
{
   void draw();
   Drawable *setColor(const color c);
   Drawable *moveX(const int x);
   Drawable *moveY(const int y);
};
Note that the methods return a Drawable object because I don't know anything about Shape. In the
Shape class, we already have implementations that are suitable for overriding these methods, because
Shape inherits from Drawable (Shape "are sort of" Drawable objects).
Now third-party developers can add other families of Drawable classes to the drawing program, in
particular, not only shapes, but also text, bitmaps, and also, amazingly, collections of other Drawables,
which allows you to nest objects in each other and make complex compositions. It is enough to inherit
from the interface and implement its methods.
class Text : public Drawable
{
public:
   Text(const string label)
   {
      ...
   }
   
   void draw()
   {
      ...
   }
   
   Text *setColor(const color c)
   {
      ...
      return &this;
   }
   ...
};
If the shape classes were distributed as a binary ex5 library (without source codes), we would supply a
header file for it containing only the description of the interface, and no hints about the internal data
structures.

---

## Page 231

Part 3. Object Oriented Programming
231 
3.2 Classes and interfaces
Since virtual functions are dynamically (later) bound to an object during program execution, it is
possible to get a "Pure virtual function call" fatal error: the program terminates. This happens if the
programmer inadvertently "forgot" to provide an implementation. The compiler is not always able to
detect such omissions at compile time.
3.2.1 9 Operator overloading
In the Expressions chapter, we learned about various operations defined for built-in types. For example,
for variables of type double, we could evaluate the following expression:
double a = 2.0, b = 3.0, c = 5.0;
double d = a * b + c;
It would be convenient to use a similar syntax when working with user-defined types, such as matrices:
Matrix a(3, 3), b(3, 3), c(3, 3); // creating 3x3 matrices
// ... somehow fill in a, b, c
Matrix d = a * b + c;
MQL5 provides such an opportunity due to operator overloading.
This technique is organized by describing methods with a name beginning with the keyword operator
and then containing a symbol (or sequence of symbols) of one of the supported operations. In a
generalized form, this can be represented as follows:
result_type operator@ ( [type parameter_name] );
Here @ - operation's symbol(s).
The complete list of MQL5 operations has been provided in the section Operation Priorities, however,
not all of them are allowed for overloading.
Forbidden for overloading:
• colons '::', context permission;
• parentheses '()', "function call" or "grouping";
• dot '.', "dereference";
• ampersand '&', "taking address", unary operator (however, the ampersand is available as binary
operator "bitwise AND");
• conditional ternary '?:';
• comma ','.
All other operators are available for overloading. Overloading operator priorities cannot be changed,
they remain equal to the standard precedence, so grouping with parentheses should be used if
necessary.
You cannot create an overload for some new character that is not included in the standard list.
All operators are overloaded taking into account their unarity and binarity, that is, the number of
required operands is preserved. Like any class method, operator overloading can return a value of some
type. In this case, the type itself should be chosen based on the planned logic of using the result of the
function in expressions (see further along).

---

## Page 232

Part 3. Object Oriented Programming
232
3.2 Classes and interfaces
Operator overloading methods have the following form (instead of the '@' symbol, the symbol(s) of the
required operator is substituted):
Name
Method header
Using 
 in an
expression
Function
is equivalent to
unary prefix
type operator@()
@object
object.operator@()
unary postfix
type operator@(int)
object@
object.operator@(0)
binary
type operator@(type parameter_name)
object@argument
object.operator@(argument)
index
type operator[](type index_name)
object[argument]
object.operator[](argument)
Unary operators do not take parameters. Of the unary operators, only the increment '++' and
decrement '--' operators support the postfix form in addition to the prefix form, all other unary
operators only support the prefix form. Specifying an anonymous parameter of type int is used to
denote the postfix form (to distinguish it from the prefix form), but the parameter itself is ignored.
Binary operators must take one parameter. For the same operator, several overloaded variants are
possible with a parameter of a different type, including the same type as the class of the current
object. In this case, objects as parameters can only be passed by reference or by pointer (the latter is
only for class objects, but not structures).
Overloaded operators can be used both via the syntax of operations as part of expressions (which is the
primary reason for overloading) and the syntax of method calls; both options are shown in the table
above. The functional equivalent makes it more obvious that technically speaking, an operator is
nothing more than a method call on an object, with the object to the right of the prefix operator and to
the left of the symbol for all others. The binary operator method will be passed as an argument the
value or expression that is to the right of the operator (this can be, in particular, another object or
variable of a built-in type).
It follows that overloaded operators do not have the commutativity property: a@b is not generally
equal to b@a, because for a the @ operator may be overloaded, but b is not. Moreover, if b is a variable
or value of a built-in type, then in principle you cannot overload the standard behavior for it.
As a first example, consider the class Fibo for generating numbers from the Fibonacci series (we have
already done one implementation of this task using functions, see Function definition). In the class, we
will provide 2 fields for storing the current and previous number of the row: current and previous,
respectively. The default constructor will initialize them with the values   1  and 0. We will also provide a
copy constructor (FiboMonad.mq5).
class Fibo
{
   int previous;
   int current;
public:
   Fibo() : current(1), previous(0) { }
   Fibo(const Fibo &other) : current(other.current), previous(other.previous) { }
   ...
};
The initial state of the object: the current number is 1 , and the previous one is 0. To find the next
number in the series, we overload the prefix and postfix increment operators.

---

## Page 233

Part 3. Object Oriented Programming
233
3.2 Classes and interfaces
   Fibo *operator++() // prefix
   {
      int temp = current;
      current = current + previous;
      previous = temp;
      return &this;
   }
   
   Fibo operator++(int) // postfix
   {
      Fibo temp = this;
      ++this;
      return temp;
   }
Please note that the prefix method does not return a pointer to the current object Fibo after the
number has been modified, but the postfix method returns to a new object with the previous counter
saved, which corresponds to the principles of postfix increment.
If necessary, the programmer, of course, can overload any operation in an arbitrary way. For example,
it is possible to calculate the product, output the number to the log, or do something else in the
implementation of the increment. However, it is recommended to stick to the approach where operator
overloading performs intuitive actions.
We implement decrement operations in a similar way: they will return the previous number of the
series.
   Fibo *operator--() // prefix
   {
      int diff = current - previous;
      current = previous;
      previous = diff;
      return &this;
   }
   
   Fibo operator--(int) // postfix
   {
      Fibo temp = this;
      --this;
      return temp;
   }
To get a number from a series by a given number, we will overload the index access operation.

---

## Page 234

Part 3. Object Oriented Programming
234
3.2 Classes and interfaces
   Fibo *operator[](int index)
   {
      current = 1;
      previous = 0;
      for(int i = 0; i < index; ++i)
      {
         ++this;
      }
      return &this;
   }
To get the current number contained in the current variable, let's overload the '~' operator (since it is
rarely used).
   int operator~() const
   {
      return current;
   }
Without this overload, you would still need to implement some public method to read the private field
current. We will use this operator to output numbers with Print.
You should also overload the assignment for convenience.
   Fibo *operator=(const Fibo &other)
   {
      current = other.current;
      previous = other.previous;
      return &this;
   }
   
   Fibo *operator=(const Fibo *other)
   {
      current = other.current;
      previous = other.previous;
      return &this;
   }
Let's check, how it all works.

---

## Page 235

Part 3. Object Oriented Programming
235
3.2 Classes and interfaces
void OnStart()
{
   Fibo f1, f2, f3, f4;
   for(int i = 0; i < 10; ++i, ++f1) // prefix increment
   {
      f4 = f3++; // postfix increment and assignment overloading
   }
   
   
// compare all values   obtained by increments and by index [10]
   Print(~f1, " ", ~f2[10], " ", ~f3, " ", ~f4); // 89 89 89 55
   
   // counting in opposite direction, down to 0
   Fibo f0;
   Fibo f = f0[10]; // copy constructor (due to initialization)
   for(int i = 0; i < 10; ++i)
   {
      // prefix decrement
      Print(~--f); // 55, 34, 21, 13, 8, 5, 3, 2, 1, 1
   }
}
The results are as expected. Still, we have to consider one detail.
   Fibo f5;
   Fibo *pf5 = &f5;
   
   f5 = f4;   // call Fibo *operator=(const Fibo &other) 
   f5 = &f4;  // call Fibo *operator=(const Fibo *other)
   pf5 = &f4; // calls nothing, assigns &f4 to pf5!
Overloading the assignment operator for a pointer only works when accessed via an object. If the
access goes via a pointer, then there is a standard assignment of one pointer to another.
The return type of an overloaded operator can be one of the built-in types, an object type (of a class or
structure), or a pointer (for class objects only).
To return an object (an instance, not a reference), the class must implement a copy constructor. This
way will cause instance duplication, which can affect the efficiency of the code. If possible, you should
return a pointer.
However, when returning a pointer, you need to make sure that it is not returning a local automatic
object (which will be deleted when the function exits, and the pointer will become invalid), but some
already existing one - as a rule, &this is returned.
Returning an object or a pointer to an object allows you to "send" the result of one overloaded operator
to another, and thereby construct complex expressions in the same way as we are accustomed to
doing with built-in types. Returning void will make it impossible to use the operator in expressions. For
example, if the '=' operator is defined with type void, then the multiple assignment will stop working:
Type x, y, z = 1; // constructors and initialization of variables of a certain class
x = y = z; // assignments, compilation error 
The assignment chain runs from right to left, and y = z will return empty.

---

## Page 236

Part 3. Object Oriented Programming
236
3.2 Classes and interfaces
If objects contain fields of built-in types only (including arrays), then the assignment/copy operator '='
from objects of the same class does not need to be redefined: MQL5 provides "one-to-one" copying of
all fields by default. The assignment/copy operator should not be confused with the copy constructor
and initialization.
Now let's turn to the second example: working with matrices(Matrix.mq5).
Note, by the way, that the built-in object types matrices and vectors have recently appeared in
MQL5. Whether to use built-in types or your own (or maybe combine them) is the choice of each
developer. Ready-made and fast implementation of many popular methods in built-in types is
convenient and eliminates routine coding. On the other hand, custom classes allow you to adapt
algorithms to your tasks. Here we provide the class Matrix as a tutorial. 
In the matrix class, we will store its elements in a one-dimensional dynamic array m. Under the sizes,
select the variables rows and columns.
class Matrix
{
protected:
   double m[];
   int rows;
   int columns;
   void assign(const int r, const int c, const double v)
   {
      m[r * columns + c] = v;
   }
      
public:
   Matrix(const Matrix &other) : rows(other.rows), columns(other.columns)
   {
      ArrayCopy(m, other.m);
   }
   
   Matrix(const int r, const int c) : rows(r), columns(c)
   {
      ArrayResize(m, rows * columns);
      ArrayInitialize(m, 0);
   }
The main constructor takes two parameters (matrix dimensions) and allocates memory for the array.
There is also a copy constructor from the other matrix other. Here and below, built-in functions for
working with arrays are massively used (in particular, ArrayCopy, ArrayResize, ArrayInitialize) – they will
be considered in a separate chapter.
We organize the filling of elements from an external array by overloading the assignment operator:

---

## Page 237

Part 3. Object Oriented Programming
237
3.2 Classes and interfaces
   Matrix *operator=(const double &a[])
   {
      if(ArraySize(a) == ArraySize(m))
      {
         ArrayCopy(m, a);
      }
      return &this;
   }
To implement the addition of two matrices, we overload the operations '+=' and '+':
   Matrix *operator+=(const Matrix &other)
   {
      for(int i = 0; i < rows * columns; ++i)
      {
         m[i] += other.m[i];
      }
      return &this;
   }
   
   Matrix operator+(const Matrix &other) const
   {
      Matrix temp(this);
      return temp += other;
   }
Note that the operator '+=' returns a pointer to the current object after it has been modified, while the
operator '+' returns a new instance by value (the copy constructor will be used), and the operator itself
has the const modifier, so how does not change the current object. 
The operator '+' is essentially a wrapper that delegates all the work to the operator '+=', having
previously created a temporary copy of the current matrix under the name temp to call it. Thus, temp
is added to other by an internal call to the operator '+=' (with temp being modified) and then returned
as the result of the ' +'.
Matrix multiplication is overloaded similarly, with two operators '*=' and '*'.

---

## Page 238

Part 3. Object Oriented Programming
238
3.2 Classes and interfaces
   Matrix *operator*=(const Matrix &other)
   {
      // multiplication condition: this.columns == other.rows
     // the result will be a matrix of size this.rows by other.columns
      Matrix temp(rows, other.columns);
      
      for(int r = 0; r < temp.rows; ++r)
      {
         for(int c = 0; c < temp.columns; ++c)
         {
            double t = 0;
            //we add up the pairwise products of the i-th elements
           // row 'r' of the current matrix and column 'c' of the matrix other
            for(int i = 0; i < columns; ++i)
            {
               t += m[r * columns + i] * other.m[i * other.columns + c];
            }
            temp.assign(r, c, t);
         }
      }
      // copy the result to the current object of the matrix this
      this = temp; // calling an overloaded assignment operator
      return &this;
   }
   
   Matrix operator*(const Matrix &other) const
   {
      Matrix temp(this);
      return temp *= other;
   }
Now, we multiply the matrix by a number:
   Matrix *operator*=(const double v)
   {
      for(int i = 0; i < ArraySize(m); ++i)
      {
         m[i] *= v;
      }
      return &this;
   }
   
   Matrix operator*(const double v) const
   {
      Matrix temp(this);
      return temp *= v;
   }
To compare two matrices, we provide the operators '==' and '!=':

---

## Page 239

Part 3. Object Oriented Programming
239
3.2 Classes and interfaces
   bool operator==(const Matrix &other) const
   {
      return ArrayCompare(m, other.m) == 0;
   }
   
   bool operator!=(const Matrix &other) const
   {
      return !(this == other);
   }
For debugging purposes, we implement the output of the matrix array to the log.
   void print() const
   {
      ArrayPrint(m);
   }
In addition to the described overloads, the class Matrix additionally has an overload of the operator []:
it returns an object of the nested class MatrixRow, i.e., a row with a given number.
   MatrixRow operator[](int r)
   {
      return MatrixRow(this, r);
   }
The class MatrixRow itself provides more "deep" access to the elements of the matrix by overloading
the same operator [] (that is, for a matrix, it will be possible to naturally specify two indexes m[i][j ]).

---

## Page 240

Part 3. Object Oriented Programming
240
3.2 Classes and interfaces
   class MatrixRow
   {
   protected:
      const Matrix *owner;
      const int row;
      
   public:
      class MatrixElement
      {
      protected:
         const MatrixRow *row;
         const int column;
         
      public:
         MatrixElement(const MatrixRow &mr, const int c) : row(&mr), column(c) { }
         MatrixElement(const MatrixElement &other) : row(other.row), column(other.column) { }
         
         double operator~() const
         {
            return row.owner.m[row.row * row.owner.columns + column];
         }
         
         double operator=(const double v)
         {
            row.owner.m[row.row * row.owner.columns + column] = v;
            return v;
         }
      };
   
      MatrixRow(const Matrix &m, const int r) : owner(&m), row(r) { }
      MatrixRow(const MatrixRow &other) : owner(other.owner), row(other.row) { }
      
      MatrixElement operator[](int c)
      {
         return MatrixElement(this, c);
      }
   
      double operator[](uint c)
      {
         return owner.m[row * owner.columns + c];
      }
   };
The operator [] for a type parameter int returns an object of class MatrixElement, through which you
can write a specific element in the array. To read an element, the operator [] is used with a type
parameter uint. This seems like a trick, but this is a language limitation: overloads must differ in the
parameter type. As an alternative to reading an element, the class MatrixElement provides an overload
of the operator '~'.
When working with matrices, you often need an identity matrix, so let's create a derived class for it:

---

## Page 241

Part 3. Object Oriented Programming
241 
3.2 Classes and interfaces
class MatrixIdentity : public Matrix
{
public:
   MatrixIdentity(const int n) : Matrix(n, n)
   {
      for(int i = 0; i < n; ++i)
      {
         m[i * rows + i] = 1;
      }
   }
};
Now let's try matrix expressions in action.
void OnStart()
{
   Matrix m(2, 3), n(3, 2); // description
   MatrixIdentity p(2);     // identity matrix
   
   double ma[] = {-1,  0, -3,
                   4, -5,  6};
   double na[] = {7,  8,
                  9,  1,
                  2,  3};
   m = ma; // filling in data
   n = na;
   
   //we can read and write elements separately
   m[0][0] = m[0][(uint)0] + 2; // variant 1 
   m[0][1] = ~m[0][1] + 2;      // variant 2 
   
   Matrix r = m * n + p;                    // expression
   Matrix r2 = m.operator*(n).operator+(p); // equivalent
   Print(r == r2); // true
   
   m.print(); // 1.00000  2.00000 -3.00000  4.00000 -5.00000  6.00000
   n.print(); // 7.00000 8.00000 9.00000 1.00000 2.00000 3.00000
   r.print(); // 20.00000  1.00000 -5.00000  46.00000
}
Here we have created 2 matrices of 3 by 2 and 2 by 3 dimensions, respectively, then filled them with
values   from arrays and edited the selective element using the syntax of two indexes [][]. Finally, we
calculated the expression m * n + p, where all operands are matrices. The line below shows the same
expression in the form of method calls. We've got the same results.
Unlike C++, MQL5 does not support operator overloading at the global level. In MQL5, an operator
can only be overloaded in the context of a class or structure, that is, using their method. Also,
MQL5 does not support overloading of type casting, operators new and delete.

---

## Page 242

Part 3. Object Oriented Programming
242
3.2 Classes and interfaces
3.2.20 Object type casting: dynamic_cast and pointer void *
Object types have specific casting rules which apply when source and destination variable types do not
match. Rules for built-in types have already been discussed in Chapter 2.6 Type conversion. The
specifics of structure type casting of structures when copying were described in the Structure layout
and inheritance section. 
For both structures and classes, the main condition for the admissibility of type casting is that they
should be related along the inheritance chain. Types from different branches of the hierarchy or not
related at all cannot be cast to each other.
Casting rules are different for objects (values) and pointers.
Objects
An object of one type A can be assigned to an object of another type B if the latter has a constructor
that takes a parameter of type A (with variations by value, reference or pointer, but usually of the form
B(const A &a)). Such a constructor is also called a conversion constructor.
In the absence of such an explicit constructor, the compiler will try to use an implicit copy operator,
i.e. B::operator=(const B &b), while classes A and B must be in the same inheritance chain for the
implicit copy to work. conversion from A to B. If A is inherited from B (including not directly, but
indirectly), then the properties added to A will disappear when copied to B. If B is inherited from A, then
only that part of the properties that are in A will be copied into it. Such conversions are usually not
welcome.
Also, the implicit copy operator may not always be provided by the compiler. In particular, if the class
has fields with the modifier const, copying is considered prohibited (see further along).
In the script ShapesCasting.mq5, we use the shape class hierarchy to demonstrate object type
conversions. In the class Shape, the field type is deliberately made constant, so an attempt to convert
(assign) an object Square to an object Rectangle ends with an error compiler with detailed explanations:
   attempting to reference deleted function 'void Rectangle::operator=(const Rectangle&)'
      function 'void Rectangle::operator=(const Rectangle&)' was implicitly deleted
      because it invokes deleted function 'void Shape::operator=(const Shape&)'
   function 'void Shape::operator=(const Shape&)' was implicitly deleted
      because member 'type' has 'const' modifier
According to this message, the copy method Rectangle::operator=(const Rectangle&) was implicitly
removed by the compiler (which provides its default implementation) because it uses a similar method
in the base class Shape::operator =(const Shape&), which in turn was removed due to the presence of
the field type with the modifier const. Such fields can only be set when the object is created, and the
compiler does not know how to copy the object under such a restriction.
By the way, the effect of "deleting" methods is available not only to the compiler but to the application
programmer: more about this will be discussed in the Inheritance control: final and delete section.
The problem could be solved by removing the modifier const or by providing your own implementation of
the assignment operator (in it, the const field is not involved and will save the content with a description
of the type: "Rectangle"):

---

## Page 243

Part 3. Object Oriented Programming
243
3.2 Classes and interfaces
   Rectangle *operator=(const Rectangle &r)
   {
      coordinates.x = r.coordinates.x;
      coordinates.y = r.coordinates.y;
      backgroundColor = r.backgroundColor;
      dx = r.dx;
      dy = r.dy;
      return &this;
   }
Note that this definition returns a pointer to the current object, while the default implementation
generated by the compiler was of type void (as seen in the error message). This means that the
compiler-provided default assignment operators cannot be used in the chain x = y = z. If you require
this capability, override operator= explicitly and return the desired type other than void.
Pointers
The most practical is to convert pointers to objects of different types.
In theory, all options for casting object type pointers can be reduced to three:
• From base to derived, the downward type casting (downcast), because it is customary to draw a
class hierarchy with an inverted tree;
• From derivative to base, the ascending type casting (upcast); and
• Between classes of different branches of the hierarchy or even from different families.
The last option is forbidden (we will get a compilation error). The compiler allows the first two, but if
"upcast" is natural and safe, then "downcast" can lead to runtime errors.
void OnStart()
{
   Rectangle *r = addRandomShape(Shape::SHAPES::RECTANGLE);
   Square *s = addRandomShape(Shape::SHAPES::SQUARE);
   Circle *c = NULL;
   Shape *p;
   Rectangle *r2;
   
   // OK
   p = c;   // Circle -> Shape
   p = s;   // Square -> Shape
   p = r;   // Rectangle -> Shape
   r2 = p;  // Shape -> Rectangle
   ...
};
Of course, when a pointer to an object of the base class is used, methods and properties of the derived
class cannot be called on it, even if the corresponding object is located at the pointer. We will get an
"undeclared identifier" compilation error.
However, the explicit cast syntax is supported for pointers (see C-style), which allows the "on the fly"
conversion of a pointer to the required type in expressions and its dereferencing without creating an
intermediate variable.

---

## Page 244

Part 3. Object Oriented Programming
244
3.2 Classes and interfaces
Base *b;
Derived d;
b = &d;
((Derived *)b).derivedMethod();
Here we have created a derived class object (Derived) and a base type pointer to it (Base *). To access
the method derivedMethod of a derived class, the pointer is temporarily converted to type Derived.
An asterisk pointer type must be enclosed in parentheses. In addition, the cast expression itself,
including the variable name, is also surrounded by another pair of parentheses.
Another compilation error ("type mismatch" - "type mismatch") in our test generates a line where we
try to cast a pointer to Rectangle to a pointer to Circle: they are from different inheritance branches.
   c = r; // error: type mismatch
Things are much worse when the type of the pointer being cast to does not match the actual object
(although their types are compatible, and therefore the program compiles fine). Such an operation will
end with an error already at the program execution stage (that is, the compiler cannot catch it). The
program is then unloaded.
For example, in the script ShapesCasting.mq5 we have described a pointer to Square and assigned it a
pointer to Shape, which contains the object Rectangle.
   Square *s2;
   // RUNTIME ERROR
   s2 = p; // error: Incorrect casting of pointers
The terminal returns the "Incorrect casting of pointers" error. The pointer of a more specific type
Square is not capable of pointing to the parent object Rectangle.
To avoid runtime troubles and to prevent the program from crashing, MQL5 provides a special language
construct dynamic_ cast. With this construct, you can "carefully" check whether it is possible to cast a
pointer to the required type. If the conversion is possible, then it will be made. And if not, we will get a
null pointer (NULL) and we can process it in a special way (for example, using if to somehow initialize or
interrupt the execution of the function, but not the entire program).
The syntax of dynamic_ cast is as follows:
dynamic_cast< Class * >( pointer )
In our case, it is enough to write:
   s2 = dynamic_cast<Square *>(p); // trying to cast type, and will get NULL if unsuccessful
   Print(s2); // 0
The program will run as expected.
In particular, we can try again to cast a rectangle into a circle and make sure that we get 0:
   c = dynamic_cast<Circle *>(r); // trying to cast type, and will get NULL if unsuccessful
   Print(c); // 0
There is a special pointer type in MQL5 that can store any object. This type has the following notation:
void *.
Let's demonstrate how the variable void * works with dynamic_ cast.

---

## Page 245

Part 3. Object Oriented Programming
245
3.2 Classes and interfaces
   void *v;
   v = s;   // set to the instance Square
   PRT(dynamic_cast<Shape *>(v));
   PRT(dynamic_cast<Rectangle *>(v));
   PRT(dynamic_cast<Square *>(v));
   PRT(dynamic_cast<Circle *>(v));
   PRT(dynamic_cast<Triangle *>(v));
The first three lines will log the value of the pointer (a descriptor of the same object), and the last two
will print 0.
Now, back to the example of the forward declaration in the Indicators section (see file
ThisCallback.mq5), where the classes Manager and Element contained mutual pointers.
The pointer type void * allows you to get rid of the preliminary declaration (ThisCallbackVoid.mq5). Let's
comment out the line with it, and change the type of the field owner with a pointer to the manager
object to void *. In the constructor, we also change the type of the parameter.
// class Manager; 
class Element
{
   void *owner; // looking forward to being compatible with the Manager type *
public:
   Element(void *t = NULL): owner(t) { } // was Element(Manager &t)
   void doMath()
   {
      const int N = 1000000;
      
      // get the desired type at runtime
      Manager *ptr = dynamic_cast<Manager *>(owner);
      // then everywhere you need to check ptr for NULL before using
      
      for(int i = 0; i < N; ++i)
      {
         if(i % (N / 20) == 0)
         {
            if(ptr != NULL) ptr.progressNotify(&this, i * 100.0f / N);
         }
         // ... lots of calculations
      }
      if(ptr != NULL) ptr.progressNotify(&this, 100.0f);
   }
   ...
};
This approach can provide more flexibility but requires more care because dynamic_ cast can return
NULL. It is recommended, whenever possible, to use standard dispatch facilities (static and dynamic)
with control of the types provided by the language.
Pointers void * usually become necessary in exceptional cases. And the "extra" line with a preliminary
description is not the case. It has been used here only as the simplest example of the universality of
the pointer void *.

---

## Page 246

Part 3. Object Oriented Programming
246
3.2 Classes and interfaces
3.2.21  Pointers, references, and const
After learning about built-in and object types, and the concepts of reference and pointer, it probably
makes sense to do a comparison of all available type modifications.
References in MQL5 are used only when describing parameters of functions and methods. Moreover,
object type parameters must be passed by reference.
void function(ClassOrStruct &object) { }          // OK
void function(ClassOrStruct object) { }           // wrong
void function(double &value) { }                  // OK
void function(double value) { }                   // OK
Here ClassOrStruct is the name of the class or structure.
It is allowed to pass only variables (LValue) as an argument for a reference type parameter, but not
constants or temporary values obtained as a result of expression evaluation.
You cannot create a variable of a reference type or return a reference from a function.
ClassOrStruct &function(void) { return Class(); } // wrong
ClassOrStruct &object;                            // wrong
double &value;                                    // wrong
Pointers in MQL5 are available only for class objects. Pointers to variables of built-in types or structures
are not supported.
You can declare a variable or function parameter of type a pointer to an object, and also return a
pointer to an object from the function.
ClassOrStruct *pointer;                                   // OK
void function(ClassOrStruct *object) { }                  // OK
ClassOrStruct *function() { return new ClassOrStruct(); } // OK
However, you cannot return a pointer to a local automatic object, because the latter will be freed when
the function exits, and the pointer will become invalid.
If the function returned a pointer to an object dynamically allocated within the function with new, then
the calling code must "remember" to free the pointer with delete.
A pointer, unlike a reference, can be NULL. Pointer parameters can have a default value, but
references can't ("reference cannot be initialized" error).
void function(ClassOrStruct *object = NULL) { }          // OK
void function(ClassOrStruct &object = NULL) { }          // wrong
Links and pointers can be combined in a parameter description. So a function can take a reference to a
pointer: and then changes to the pointer in the function will become available in the calling code. In
particular, the factory function, which is responsible for creating objects, can be implemented in this
way.

---

## Page 247

Part 3. Object Oriented Programming
247
3.2 Classes and interfaces
void createObject(ClassName *&ref)
{
   ref = new ClassName();
   // further customization of ref
   ...
}
True, to return a single pointer from a function, it is usually customary to use the return statement, so
this example is somewhat artificial. However, in those cases when it is necessary to pass an array of
pointers outside, a reference to it in the parameter becomes the preferred option. For example, in
some classes of the standard library for working with container classes of the map type with [key,
value] pairs (MQL5/Include/Generic/SortedMap.mqh, MQL5/Include/Generic/HashMap.mqh) there are
methods CopyTo for getting arrays with elements CKeyValuePair.
int CopyTo(CKeyValuePair<TKey,TValue> *&dst_array[], const int dst_start = 0);
The parameter type dst_ array may seem unfamiliar: it's a class template. We will learn about
templates in the next chapter. Here, for now, the only important thing for us is that this is a reference
to an array of pointers.
The const modifier imposes special behavior for all types. In relation to built-in types, it was discussed
in the section on Constant variables. Object types have their own characteristics.
If a variable or function parameter is declared as a pointer or a reference to an object (a reference is
only in the case of a parameter), then the presence of the modifier const on them limits the set of
methods and properties that can be accessed to only those that also have the modifier const. In other
words, only constant properties are accessible through constant references and pointers.
When you try to call a non-const method or change a non-const field, the compiler will generate an
error: "call non-const method for constant object" or "constant cannot be modified".
A non-const pointer parameter can take any argument (constant or non-constant).
It should be borne in mind that two modifiers const can be set in the pointer description: one will refer
to the object, and the second to the pointer:
• Class *pointer is a pointer to an object; the object and the pointer work without limitations;
• const Class *pointer is a pointer to a const object; for the object, only constant methods and
reading properties are available, but the pointer can be changed (assigned to it the address of
another object);
• const Class * const pointer is a const pointer to a const object; for the object, only const methods
and reading properties are available; the pointer cannot be changed;
• Class * const pointer is a const pointer to an object; the pointer cannot be changed, but the
properties of the object can be changed.
Consider the following class Counter (CounterConstPtr.mq5) as an example.

---

## Page 248

Part 3. Object Oriented Programming
248
3.2 Classes and interfaces
class Counter
{
public:
   int counter;
   
   Counter(const int n = 0) : counter(n) { }
   
   void increment()
   {
      ++counter;
   }
   
   Counter *clone() const
   {
      return new Counter(counter);
   }
};
It artificially made the public variable counter. The class also has two methods, one of which is
constant (clone), and the second is not (increment). Recall that a constant method does not have the
right to change the fields of an object.
The following function with the Counter *ptr type parameter can call all methods of the class and
change its fields.
void functionVolatile(Counter *ptr)
{
   // OK: everything is available
   ptr.increment();
   ptr.counter += 2;
   //remove the clone immediately so that there is no memory leak
   // the clone is only needed to demonstrate calling a constant method 
   delete ptr.clone(); 
   ptr = NULL;
}
The following function with the parameter const Counter *ptr will throw a couple of errors.
void functionConst(const Counter *ptr)
{
   // ERRORS:
   ptr.increment(); // calling non-const method for constant object
   ptr.counter = 1; // constant cannot be modified
   
   // OK: only const methods are available, fields can be read
   Print(ptr.counter); // reading a const object
   Counter *clone = ptr.clone(); // calling a const method
   ptr = clone;     // changing a non-const pointer ptr
   delete ptr;      // cleaning memory
}
Finally, the following function with the parameter const Counter * const ptr does even less.

---

## Page 249

Part 3. Object Oriented Programming
249
3.2 Classes and interfaces
void functionConstConst(const Counter * const ptr)
{
   // OK: only const methods are available, the pointer ptr cannot be changed
   Print(ptr.counter); // reading a const object
   delete ptr.clone(); // calling a const method
   
   Counter local(0);
   // ERRORS:
   ptr.increment(); // calling non-const method for constant object
   ptr.counter = 1; // constant cannot be modified
   ptr = &local;    // constant cannot be modified
}
In the function OnStart, where we have declared two Counter objects (one is constant and the other is
not), you can call these functions with some exceptions:
void OnStart()
{
   Counter counter;
   const Counter constCounter;
   
   counter.increment();
   
   // ERROR:
   // constCounter.increment(); // call non-const method for constant object
   Counter *ptr = (Counter *)&constCounter; // trick: type casting without const
   ptr.increment();
   
   functionVolatile(&counter);
   
   // ERROR: cannot convert from a const pointer...
   // functionVolatile(&constCounter); // to a non-const pointer
   
   functionVolatile((Counter *)&constCounter); // type casting without const
   
   functionConst(&counter);
   functionConst(&constCounter);
   
   functionConstConst(&counter);
   functionConstConst(&constCounter);
}
First, note that variables also generate an error when trying to call a const method increment on a non-
const object.
Secondly, constCounter cannot be passed to the functionVolatile function – we get the error "cannot
convert from const pointer to nonconst pointer".
However, both errors can be circumvented by explicit type casting without the const modifier. Although
this is not recommended.

---

## Page 250

Part 3. Object Oriented Programming
250
3.2 Classes and interfaces
3.2.22 Inheritance management: final and delete
MQL5 allows you to impose some restrictions on the inheritance of classes and structures.
Keyword final
By using the final keyword added after the class name, the developer can disable inheritance from that
class. For example (FinalDelete.mq5):
class Base
{
};
class Derived final : public Base
{
};
class Concrete : public Derived // ERROR
{
};
The compiler will throw the error "cannot inherit from 'Derived' as it has been declared as 'final'".
Unfortunately, there is no consensus on the benefits and scenarios for using such a restriction. The
keyword lets users of the class know that its author, for one reason or another, does not recommend
taking it as the base one (for example, its current implementation is draft and will change a lot, which
may cause potential legacy projects to stop compiling).
Some people try to encourage the design of programs in this way, in which the inclusion of objects
(composition) is used instead of inheritance. Excessive passion for inheritance can indeed increase the
class cohesion (that is, mutual influence), since all heirs in one way or another can change parent data
or methods (in particular, by redefining virtual functions). As a result, the complexity of the working
logic of the program and the likelihood of unforeseen side effects increase.
An additional advantage of using final can be code optimization by the compiler: for pointers of "final"
types, it can replace the dynamic dispatch of virtual functions with a static one.
Keyword delete
The delete keyword can be specified in the header of a method to make it inaccessible in the current
class and its descendants. Virtual methods of parent classes cannot be deleted in this way (this would
violate the "contract" of the class, that is, the heirs would cease to "be" ("is a") representatives of the
same kind).

---

## Page 251

Part 3. Object Oriented Programming
251 
3.2 Classes and interfaces
class Base
{
public:
   void method() { Print(__FUNCSIG__); }
};
class Derived : public Base
{
public:
   void method() = delete;
};
void OnStart()
{
   Base *b;
   Derived d;
   
   b = &d;
   b.method();
   
   // ERROR:   
   // attempting to reference deleted function 'void Derived::method()'
   //    function 'void Derived::method()' was explicitly deleted
   d.method();
}
An attempt to call it will result in a compilation error.
We saw a similar error in the Object type casting section because the compiler has some intelligence to
also "remove" methods under certain conditions.
It is recommended to mark as deleted the following methods for which the compiler provides implicit
implementations:
• default constructor: Class(void) = delete;
• copy constructor: Class(const Class &obj ect) = delete;
• copy/assign operator: void operator=(const Class &obj ect) = delete.
If you require any of these, you must define them explicitly. Otherwise, it is considered good practice
to abandon the implicit implementation. The thing is that the implicit implementation is quite
straightforward and can give rise to problems that are difficult to localize, in particular, when casting
object types.
3.3 Templates
In modern programming languages, there are many built-in features that allow you to avoid code
duplication and, thereby, minimize the number of errors and increase programmer productivity. In
MQL5, such tools include the already known functions, object types with inheritance support (classes
and structures), preprocessor macros, and the ability to include files. But this list would be incomplete
without templates.

---

## Page 252

Part 3. Object Oriented Programming
252
3.3 Templates
A template is a specially crafted generic definition of a function or object type from which the compiler
can automatically generate working instances of that function or object type. The resulting instances
contain the same algorithm but operate on variables of different types, corresponding to the specific
conditions for using the template in the source code.
For C++ connoisseurs, we note that MQL5 templates do not support many features of C++ templates,
in particular:
·parameters that are not types;
·parameters with default values;
·variable number of parameters;
·specialization of classes, structures, and associations (full and partial);
·templates for templates.
On the one hand, this reduces the potential of templates in MQL5, but, on the other hand, it simplifies
the learning of the material for those who are unfamiliar with these technologies.
3.3.1  Template header
In MQL5, you can make functions, object types (classes, structures, unions) or separate methods
within them templated. In any case, the template description has a title:
template <typename T [, typename Ti ... ]>
The header starts with the template keyword, followed by a comma-separated list of formal parameters
in angle brackets: each parameter is denoted by the typename keyword and an identifier. Identifiers
must be unique within a particular definition.
The keyword typename in the template header tells the compiler that the following identifier should be
treated as a type. In the future, the MQL5 compiler is likely to support other kinds of non-type
parameters, as the C++ compiler does.
This use of typename should not be confused with the built-in operator typename, which returns a
string with the type name of the passed argument.
A template header is followed by a usual definition of a function (method) or class (structure, union), in
which the formal parameters of the template (identifiers T, Ti) are used in instructions and expressions
in those places where the syntax requires a type name. For example, for template functions, template
parameters describe the types of the function parameters or return value, and in a template class, a
template parameter can designate a field type.
A template is an entire definition. A template ends with a definition of an entity (function, method,
class, structure, union) preceded by the template heading.
For template parameter names, it is customary to take one- or two-character identifiers in uppercase.
The minimum number of parameters is 1 , the maximum is 64.
The main use cases for parameters (using the T parameter as an example) include:
• type when describing fields, local variables in functions/methods, their parameters and return
values   (T variable_ name; T function(T parameter_ name));
• one of the components of a fully qualified type name, in particular: T::SubType, T.StaticMember;

---

## Page 253

Part 3. Object Oriented Programming
253
3.3 Templates
• construction of new types with modifiers: const T, pointer T *, reference T &, array T[], typedef
functions T(*func)(T);
• construction of new template types: T<Type>, Type<T>, including when inheriting from templates
(see section Template specialization, which is not present);
• typecasting (T) with the ability to add modifiers and creating objects via new T();
• sizeof(T) as a primitive replacement for value parameters that are absent in MQL templates (at the
time of writing the book).
3.3.2 General template operation principles
Let's recall functions overload. It consists in defining several versions of a function with different
parameters, including situations when the number of parameters is the same, but their types are
different. Often an algorithm of such functions is the same for parameters of different types. For
example, MQL5 has a built-in function MathMax that returns the largest of the two values passed to it:
double MathMax(double value1, double value2);
Although a prototype is only provided for the type double, the function is actually capable of working
with argument pairs of other numeric types, such as int or datetime. In other words, the function is an
overloaded kernel for built-in numerical types. If we wanted to achieve the same effect in our source
code, we would have to overload the function by duplicating it with different parameters, like so:
double Max(double value1, double value2)
{
   return value1 > value2 ? value1 : value2;
}
int Max(int value1, int value2)
{
   return value1 > value2 ? value1 : value2;
}
datetime Max(datetime value1, datetime value2)
{
   return value1 > value2 ? value1 : value2;
}
All implementations (function bodies) are the same. Only the parameter types change.
This is when templates are useful. By using them, we can describe one sample of the algorithm with the
required implementation, and the compiler itself will generate several instances of it for the specific
types involved in the program. Generation occurs on the fly during compilation and is imperceptible to
the programmer (unless there is an error in the template) The source code obtained automatically is
not inserted into the program text, but is directly converted into binary code (ex5 file).
In the template, one or more parameters are formal designations of types, for which, at the
compilation stage, according to special type inference rules, real types will be selected from among
built-in or user-defined ones. For example, the Max function can be described using the following
template with the T type parameter:

---

## Page 254

Part 3. Object Oriented Programming
254
3.3 Templates
template<typename T>
T Max(T value1, T value2)
{
   return value1 > value2 ? value1 : value2;
}
And then - apply it for variables of various types (see TemplatesMax.mq5):
void OnStart()
{
   double d1 = 0, d2 = 1;
   datetime t1 = D'2020.01.01', t2 = D'2021.10.10';
   Print(Max(d1, d2));
   Print(Max(t1, t2));
   ...
}
In this case, the compiler will automatically generate variants of the function Max for the types double
and datetime.
The template itself does not generate source code. To do this, you need to create an instance of the
template in one way or another: call a template function or mention the name of a template class with
specific types to create an object or a derived class.
Until this is done, the entire pattern is ignored by the compiler. For example, we can write the following
supposedly template function, which actually contains syntactically incorrect code. However, the
compilation of a module with this function will succeed as long as it is not called anywhere.
template<typename T>
void function()
{
  it's not a comment, but it's not source code either
   !%^&*
}
For each use of the template, the compiler determines the real types that match the formal
parameters of the template. Based on this information, template source code is automatically
generated for each unique combination of parameters. This is the instance.
So, in the given example of the Max function, we called the template function twice: for the pair of
variables of type double, and for the pair of variables of type datetime. This resulted in two instances of
the Max function with source code for the matches T=double and T=datetime. Of course, if the same
template is called in other parts of the code for the same types, no new instances will be generated. A
new instance of the template is required only if the template is applied to another type (or set of types,
if there is more than 1  parameter).
Please note that the template Max has one parameter, and it sets the type for two input parameters of
the function and its return value at once. In other words, the template declaration is capable of
imposing certain restrictions on the types of valid arguments.
If we were to call Max on variables of different types, the compiler would not be able to determine the
type to instantiate the template and would throw the error "ambiguous template parameters, must be
'double' or 'datetime'":

---

## Page 255

Part 3. Object Oriented Programming
255
3.3 Templates
Print(Max(d1, t1)); // template parameter ambiguous,
                    // could be 'double' or 'datetime'
This process of discovering the actual types for template parameters based on the context in which the
template is used is called type deduction. In MQL5, type inference is available only for function and
method templates.
For classes, structures, and unions, a different way of binding types to template parameters is used:
the required types are explicitly specified in angle brackets when creating a template instance (if there
are several parameters, then the corresponding number of types is indicated as a comma-separated
list). For more on this, see the section Object type templates.
The same explicit method can be applied to functions as an alternative to automatic type inference.
For example, we can generate and call an instance of Max for type ulong:
Print(Max<ulong>(1000, 10000000));
In this case, if not for the explicit indication, the template function would be associated with the type
int 
(based on the values   of integer constants).
3.3.3 Templates vs preprocessor macros
A question may arise at some point, is it possible to use macro substitutions for the purposes of code
generation? It is actually possible. For example, the set of Max functions can be easily represented as a
macro:
#define MAX(V1,V2) ((V1) > (V2) ? (V1) : (V2))
However, macros have more limited capabilities (nothing more than text substitution) and therefore
they are only used in simple cases (like the one above).
When comparing macros and templates, the following differences should be noted.
Macros are "expanded" and replaced in the source text by the preprocessor before compilation starts.
At the same time, there is no information about the types of parameters and the context in which the
contents of the macro are substituted. In particular, the macro MAX cannot provide a check-up that
the types of the parameters V1  and V2 are the same, and also that the comparison operator '>' is
defined for them. In addition, if a variable with the name MAX is encountered in a program text, the
preprocessor will try to substitute the "call" of the MAX macro in its place and will be "unhappy" with
the absence of arguments. Worse yet, these substitutions ignore which namespaces or classes the MAX
token is found in — basically, any will do.
Unlike macros, templates are handled by the compiler in terms of specific argument types and where
they are used, so they provide type compatibility (and general applicability) checks for all expressions
in a template, as well as context binding. For example, we can define a method template within a
concrete class.
A template with the same name can be defined differently for different types if necessary, while a
macro with a given name is always replaced by the same "implementation". For example, in the case of
a function like MAX, we could define a case-insensitive comparison for strings.
Compilation errors due to problems in macros are difficult to diagnose, especially if the macro consists
of several lines, since the problematic line with the "call" of the macro is highlighted "as is", without the
expanded version of the text, as it came from the preprocessor to the compiler.

---

## Page 256

Part 3. Object Oriented Programming
256
3.3 Templates
At the same time, templates are elements of the source code in a ready-made form, as they enter the
compiler, and therefore any error in them has a specific line number and position in the line.
Macros can have side effects, which we discussed in the Form of #define as a pseudo-function section:
if the MAX macro arguments are expressions with increments/decrements, then they will be executed
twice.
However, macros also have some advantages. Macros are capable of generating any text, not just
correct language constructs. For example, with a few macros, you can simulate the instruction switch
for strings (although this approach is not recommended).
In the standard library, macros are used, in particular, to organize the processing of events on charts
(see MQL5/Include/Controls/Defines.mqh: EVENT_MAP_BEGIN, EVENT_MAP_END, ON_EVENT, etc.). It
will not work on templates, but the way of arranging an event map on macros, of course, is far from the
only one and not the most convenient for debugging. It is difficult to debug step-by-step (line-by-line)
code execution in macros. Templates, on the contrary, support debugging in full.
3.3.4 Features of built-in and object types in templates
It should be kept in mind that 3 important aspects impose restrictions on the applicability of types in a
template:
• Whether the type is built-in or user-defined (user-defined types require parameters to be passed by
reference, and built-in ones will not allow a literal to be passed by reference);
• Whether the object type is a class (only classes support pointers);
• A set of operations performed on data of the appropriate types in the template algorithm.
Let's say we have a Dummy structure (see script TemplatesMax.mq5):
struct Dummy
{
   int x;
};
If we try to call the Max function for two instances of the structure, we will get a bunch of error
messages, with mains as the following: "objects can only be passed by reference" and "you cannot
apply a template."
   // ERRORS:
   // 'object1' - objects are passed by reference only
   // 'Max' - cannot apply template
   Dummy object1, object2;
   Max(object1, object2);
The pinnacle of the problem is passing template function parameters by value, and this method is
incompatible with any object type. To solve it, you can change the type of parameters to links:

---

## Page 257

Part 3. Object Oriented Programming
257
3.3 Templates
template<typename T>
T Max(T &value1, T &value2)
{
   return value1 > value2 ? value1 : value2;
}
The old error will go away, but then we will get a new error: "'>' - illegal operation use" ("'>' - illegal
operation use"). The point is that the Max template has an expression with the '>' comparison
operator. Therefore, if a custom type is substituted into the template, the '>' operator must be
overloaded in the template (and the structure Dummy does not have it: we'll get to that shortly). For
more complex functions, you will likely need to overload a much larger number of operators.
Fortunately, the compiler tells you exactly what is missing.
However, changing the method of passing function parameters by reference additionally led to the
previous call not working as such:
Print(Max<ulong>(1000, 10000000));
Now it generates errors: "parameter passed as reference, variable expected". Thus, our function
template stopped working with literals and other temporary values   (in particular , it is impossible to
directly pass an expression or the result of calling another function into it).
One might think that the universal way out of the situation would be template function overloading, i.e.,
the definition of both options, that differs only in the ampersand in the parameters:
template<typename T>
T Max(T &value1, T &value2)
{
   return value1 > value2 ? value1 : value2;
}
  
template<typename T>
T Max(T value1, T value2)
{
   return value1 > value2 ? value1 : value2;
}
But it won't work. Now the compiler throws the error "ambiguous function overload with the same
parameters":
'Max' - ambiguous call to overloaded function with the same parameters
could be one of 2 function(s)
   T Max(T&,T&)
   T Max(T,T)
The final, working overload would require the modifier const to be added to the links. Along the way, we
added the operator Print to the template Max so that we can see in the log which overload is being
called and which parameter type T corresponds to.

---

## Page 258

Part 3. Object Oriented Programming
258
3.3 Templates
template<typename T>
T Max(const T &value1, const T &value2)
{
   Print(__FUNCSIG__, " T=", typename(T));
   return value1 > value2 ? value1 : value2;
}
   
template<typename T>
T Max(T value1, T value2)
{
   Print(__FUNCSIG__, " T=", typename(T));
   return value1 > value2 ? value1 : value2;
}
   
struct Dummy
{
   int x;
   bool operator>(const Dummy &other) const
   {
      return x > other.x;
   }
};
We have also implemented an overload of the operator '>' in the Dummy structure. Therefore, all Max
function calls in the test script are completed successfully: both for built-in and user-defined types, as
well as for literals and variables. The outputs that go into the log:
double Max<double>(double,double) T=double
1.0
datetime Max<datetime>(datetime,datetime) T=datetime
2021.10.10 00:00:00
ulong OnStart::Max<ulong>(ulong,ulong) T=ulong
10000000
Dummy Max<Dummy>(const Dummy&,const Dummy&) T=Dummy
An attentive reader will notice that we now have two identical functions that differ only in the way
parameters are passed (by value and by reference), and this is exactly the situation against which the
use of templates is directed. Such duplication can be costly if the function body is not as simple as
ours. This can be solved by the usual methods: separate the implementation into a separate function
and call it from both "overloads", or call one "overload" from the other (an optional parameter was
required to avoid the first version of Max calling itself and, resulting in stack overflows):

---

## Page 259

Part 3. Object Oriented Programming
259
3.3 Templates
template<typename T>
T Max(T value1, T value2)
{
   // calling a function with parameters by reference
   return Max(value1, value2, true);
}
   
template<typename T>
T Max(const T &value1, const T &value2, const bool ref = false)
{
   return (T)(value1 > value2 ? value1 : value2);
}
We still have to consider one more point associated with user-defined types, namely the use of pointers
in templates (recall, that they apply only to class objects). Let's create a simple class Data and try to
call the template function Max for pointers to its objects.
class Data
{
public:
   int x;
   bool operator>(const Data &other) const
   {
      return x > other.x;
   }
};
   
void OnStart()
{
   ... 
   Data *pointer1 = new Data();
   Data *pointer2 = new Data();
   Max(pointer1, pointer2);
   delete pointer1;
   delete pointer2;
}
We will see in the log that 'T=Data*', i.e. the pointer attribute, hits the inline type. This suggests that,
if necessary, you can write another overload of the template function, which will be responsible only for
pointers.
template<typename T>
T *Max(T *value1, T *value2)
{
   Print(__FUNCSIG__, " T=", typename(T));
   return value1 > value2 ? value1 : value2;
}
In this case, the attribute of the pointer '*' is already present in the template parameters, and so type
inference results in 'T=Data'. This approach allows you to provide a separate template implementation
for pointers.

---

## Page 260

Part 3. Object Oriented Programming
260
3.3 Templates
If there are multiple templates that are suitable for generating an instance with specific types, the
most specialized version of the template is chosen. In particular, when calling the function Max with
pointer arguments, two templates with parameters T (T=Data*) and T* (T=Data), but since the former
can take both values   and pointers, it is more general than the latter , which only works with pointers.
Therefore, the second one will be chosen for pointers. In other words, the fewer modifiers in the actual
type that is substituted for T, the more preferable the template variant. In addition to the attribute of
the pointer '*', this also includes the modifier const. The parameters const T* or const T are more
specialized than just T* or T, respectively.
3.3.5 Function templates
A function template consists of a header with template parameters (the syntax was described earlier)
and a function definition in which the template parameters denote arbitrary types.
As a first example, consider the function Swap for swapping two array elements
(TemplatesSorting.mq5). The template parameter T is used as the type of the input array variable, as
well as the type of the local variable temp.
template<typename T>
void Swap(T &array[], const int i, const int j)
{
   const T temp = array[i];
   array[i] = array[j];
   array[j] = temp;
}
All statements and expressions in the body of the function must be applicable to real types, for which
the template will then be instantiated. In this case, the assignment operator '=' is used. While it always
exists for built-in types, it may need to be explicitly overloaded for user-defined types.
The compiler generates the implementation of the copy operator for classes and structures by default,
but it can be removed implicitly or explicitly (see keyword delete). In particular, as we saw in the
section Object Type Casting, having a constant field in a class causes the compiler to remove its
implicit copy option. Then the above template function Swap cannot be used for objects of this class:
the compiler will generate an error.
For classes/structures that the Swap function works with, it is desirable to have not only an assignment
operator but also a copy constructor, because the declaration of the variable temp is actually a
construction with an initialization, not an assignment. With a copy constructor, the first line of the
function is executed in one go (temp is created based on array[i]), while without it, the default
constructor will be called first, and then for temp the operator '=' will be executed.
Let's see how the template function Swap can be used in the quicksort algorithm: another template
function QuickSort implements it.

---

## Page 261

Part 3. Object Oriented Programming
261 
3.3 Templates
template<typename T>
void QuickSort(T &array[], const int start = 0, int end = INT_MAX)
{
   if(end == INT_MAX)
   {
      end = start + ArraySize(array) - 1;
   }
   if(start < end)
   {
      int pivot = start;
      
      for(int i = start; i <= end; i++)
      {
         if(!(array[i] > array[end]))
         {
            Swap(array, i, pivot++);
         }
      }
      
      --pivot;
   
      QuickSort(array, start, pivot - 1);
      QuickSort(array, pivot + 1, end);
   }
}
Note that the T parameter of the QuickSort template specifies the type of the input parameter array,
and this array is then passed to the Swap template. Thus, type inference T for the QuickSort template
will automatically determine the type T for the Swap template.
The built-in function ArraySize (like many others) is able to work with arrays of arbitrary types: in a
sense, it is also a template, although it is implemented directly in the terminal.
Sorting is done thanks to the '>' comparison operator in the if statement. As we noted earlier, this
operator must be defined for any type T that is being sorted, because it applies to the elements of an
array of type T.
Let's check how sorting works for arrays of built-in types.

---

## Page 262

Part 3. Object Oriented Programming
262
3.3 Templates
void OnStart()
{
   double numbers[] = {34, 11, -7, 49, 15, -100, 11};
   QuickSort(numbers);
   ArrayPrint(numbers);
   // -100.00000 -7.00000 11.00000 11.00000 15.00000 34.00000 49.00000
   
   string messages[] = {"usd", "eur", "jpy", "gbp", "chf", "cad", "aud", "nzd"};
   QuickSort(messages);
   ArrayPrint(messages);
   // "aud" "cad" "chf" "eur" "gbp" "jpy" "nzd" "usd"
}
Two calls to the template function QuickSort automatically infer the type of T based on the types of the
passed arrays. As a result, we will get two instances of QuickSort for types double and string.
To check the sorting of a custom type, let's create an ABC structure with an integer field x, and fill it
with random numbers in the constructor. It is also important to overload the operator '>' in the
structure.

---

## Page 263

Part 3. Object Oriented Programming
263
3.3 Templates
struct ABC
{
   int x;
   ABC()
   {
      x = rand();
   }
   bool operator>(const ABC &other) const
   {
      return x > other.x;
   }
};
void OnStart()
{
   ...
   ABC abc[10];
   QuickSort(abc);
   ArrayPrint(abc);
  /* Sample output:
            [x]
      [0]  1210
      [1]  2458
      [2] 10816
      [3] 13148
      [4] 15393
      [5] 20788
      [6] 24225
      [7] 29919
      [8] 32309
      [9] 32589
   */
}
Since the structure values are randomly generated, we will get different results, but they will always be
sorted in ascending order.
In this case, the type T is also automatically inferred. However, in some cases, explicit specification is
the only way to pass a type to a function template. So, if a template function must return a value of a
unique type (different from the types of its parameters) or if there are no parameters, then it can only
be specified explicitly.
For example, the following template function createInstance requires the type to be explicitly specified
in the calling instruction, since it is not possible to automatically "calculate" the type T from the return
value. If this is not done, the compiler generates a "template mismatch" error.

---

## Page 264

Part 3. Object Oriented Programming
264
3.3 Templates
class Base
{
   ...
};
   
template<typename T>
T *createInstance()
{
   T *object = new T(); //calling the constructor
   ...                  //object setting
   return object; 
}
   
void OnStart()
{
   Base *p1 = createInstance();       // error: template mismatch
   Base *p2 = createInstance<Base>(); // ok, explicit directive
   ...
}
If there are several template parameters, and the type of the return value is not bound to any of the
input parameters of the function, then you also need to specify a specific type when calling:
template<typename T,typename U>
T MyCast(const U u)
{
   return (T)u;
}
   
void OnStart()
{
   double d = MyCast<double,string>("123.0");
   string f = MyCast<string,double>(123.0);
}
Note that if the types for the template are explicitly specified, then this is required for all parameters,
even though the second parameter U could be inferred from the passed argument.
After the compiler has generated all instances of the template function, they participate in the
standard procedure for choosing the best candidate from all function overloads with the same name
and the appropriate number of parameters. Of all the overload options (including the created template
instances), the closest one in terms of types (with the least number of conversions) is selected.
If a template function has some input parameters of specific types, then it is considered a candidate
only if these types completely match the arguments: any need for conversion will cause the template
to be "discarded" as unsuitable.
Non-template overloads take precedence over template overloads, more specialized ("narrowly
focused") "win" from template overloads.
If the template argument (type) is specified explicitly, then the rules for implicit type casting are
applied for the corresponding function argument (passed value), if necessary, if these types differ.

---

## Page 265

Part 3. Object Oriented Programming
265
3.3 Templates
If several variants of a function match equally, we will get an "ambiguous call to an overloaded function
with the same parameters" error.
For example, if in addition to the template MyCast, a function is defined to convert a string to a boolean
type:
bool MyCast(const string u)
{
   return u == "true";
}
then calling MyCast<double,string>("1 23.0") will start throwing the indicated error, because the two
functions differ only in the return value:
'MyCast<double,string>' - ambiguous call to overloaded function with the same parameters
could be one of 2 function(s)
   double MyCast<double,string>(const string)
   bool MyCast(const string)
When describing template functions, it is recommended to include all template parameters in the
function parameters. Types can only be inferred from arguments, not from the return value.
If a function has a templated type parameter T with a default value, and the corresponding argument is
omitted when called, then the compiler will also fail to infer the type of T and throw a "cannot apply
template" error.
class Base
{
public:
   Base(const Base *source = NULL) { }
   static Base *type;
};
   
static Base* Base::type;
   
template<typename T>
T *createInstanceFrom(T *origin = NULL)
{
   T *object = new T(origin);
   return object; 
}
   
void OnStart()
{
   Base *p1 = createInstanceFrom();   // error: cannot to apply template
   Base *p2 = createInstanceFrom(Base::type); // ok, auto-detect from argument
   Base *p3 = createInstanceFrom<Base>();     // ok, explicit directive, an argument is omitted
}
3.3.6 Object type templates
An object type template definition begins with a header containing typed parameters (see section
Template Header), and the usual definition of a class, structure, or union.

---

## Page 266

Part 3. Object Oriented Programming
266
3.3 Templates
template <typename T [, typename Ti ...] >
class class_name
{
   ...
};
The only difference from the standard definition is that template parameters can occur in a block of
code, in all syntactic constructs of the language, where it is permissible to use a type name.
Once a template is defined, working instances of it are created when the variables of the template type
are declared in the code, specifying the specific types in angle brackets:
ClassName<Type1,Type2> object;
StructName<Type1,Type2,Type3> struct;
ClassName<Type1,Type2> *pointer = new ClassName<Type1,Type2>();
ClassName1<ClassName2<Type>> object;
Unlike when calling template functions, the compiler is not able to infer actual types for object
templates on its own.
Declaring a template class/structure variable is not the only way to instantiate a template. An instance
is also generated by the compiler if a template type is used as the base type for another, specific (non-
template) class or structure.
For example, the following class Worker, even if empty, is an implementation of Base for type double:
class Worker : Base<double>
{
};
This minimum definition is enough (with allowance for adding constructors if the class Base requires
them) to start compiling and validating the template code.
In the Dynamic object creation section, we got acquainted with the concept of a dynamic pointer to an
object obtained using the operator new. This flexible mechanism has one drawback: pointers need to be
monitored and "manually" deleted when they are no longer needed. In particular, when exiting a
function or block of code, all local pointers must be cleared with a call delete.
To simplify the solution to this problem, let's create a template class AutoPtr (TemplatesAutoPtr.mq5,
AutoPtr.mqh). Its parameter T is used to describe the field ptr, which stores a pointer to an object of
an arbitrary class. We will receive the pointer value through the constructor parameter (T *p) or in the
overloaded operator '='. Let's entrust the main work to the destructor: in the destructor, the pointer
will be deleted together with the object AutoPtr (the static helper method free is allocated for this).
The principle of operation of AutoPtr is simple: a local object of this class will be automatically
destroyed upon exiting the block where it is described, and if it was previously instructed to "follow"
some pointer, then AutoPtr will free it too.

---

## Page 267

Part 3. Object Oriented Programming
267
3.3 Templates
template<typename T>
class AutoPtr
{
private:
   T *ptr;
   
public:
   AutoPtr() : ptr(NULL) { }
   
   AutoPtr(T *p) : ptr(p)
   {
      Print(__FUNCSIG__, " ", &this, ": ", ptr);
   }
   
   AutoPtr(AutoPtr &p)
   {
      Print(__FUNCSIG__, " ", &this, ": ", ptr, " -> ", p.ptr);
      free(ptr);
      ptr = p.ptr;
      p.ptr = NULL;
   }
   
   ~AutoPtr()
   {
      Print(__FUNCSIG__, " ", &this, ": ", ptr);
      free(ptr);
   }
   
   T *operator=(T *n)
   {
      Print(__FUNCSIG__, " ", &this, ": ", ptr, " -> ", n);
      free(ptr);
      ptr = n;
      return ptr;
   }
   
   T* operator[](int x = 0) const
   {
      return ptr;
   }
   
   static void free(void *p)
   {
      if(CheckPointer(p) == POINTER_DYNAMIC) delete p;
   }
};
Additionally, the class AutoPtr implements a copy constructor (more precisely, a jump constructor,
since the current object becomes the owner of the pointer), which allows you to return an AutoPtr
instance along with a controlled pointer from a function.
To test the performance of AutoPtr, we will describe a fictitious class Dummy.

---

## Page 268

Part 3. Object Oriented Programming
268
3.3 Templates
class Dummy
{
   int x;
public:
   Dummy(int i) : x(i)
   {
      Print(__FUNCSIG__, " ", &this);
   }
   ...
   int value() const
   {
      return x;
   }
};
In the script, in the OnStart function, enter the variable AutoPtr<Dummy> and get the value for it from
the function generator. In the function generator itself, we will also describe the object
AutoPtr<Dummy> and sequentially create and "attach" two dynamic objects Dummy to it (to check the
correct release memory from the "old" object).
AutoPtr<Dummy> generator()
{
   AutoPtr<Dummy> ptr(new Dummy(1));
   // pointer to 1 will be freed after execution of '='
   ptr = new Dummy(2);
   return ptr;
}
   
void OnStart()
{
   AutoPtr<Dummy> ptr = generator();
   Print(ptr[].value());             // 2
}
Since all the main methods log object descriptors (both AutoPtr and controlled pointers ptr), we can
track all "transformations" of pointers (for convenience, all lines are numbered).
01 Dummy::Dummy(int) 3145728
02  AutoPtr<Dummy>::AutoPtr<Dummy>(Dummy*) 2097152: 3145728
03  Dummy::Dummy(int) 4194304
04  Dummy*AutoPtr<Dummy>::operator=(Dummy*) 2097152: 3145728 -> 4194304
05  Dummy::~Dummy() 3145728
06  AutoPtr<Dummy>::AutoPtr<Dummy>(AutoPtr<Dummy>&) 5242880: 0 -> 4194304
07  AutoPtr<Dummy>::~AutoPtr<Dummy>() 2097152: 0
08  AutoPtr<Dummy>::AutoPtr<Dummy>(AutoPtr<Dummy>&) 1048576: 0 -> 4194304
09  AutoPtr<Dummy>::~AutoPtr<Dummy>() 5242880: 0
10  2
11  AutoPtr<Dummy>::~AutoPtr<Dummy>() 1048576: 4194304
12  Dummy::~Dummy() 4194304
Let's digress for a moment from the templates and describe in detail how the utility works because
such a class can be useful to many.

---

## Page 269

Part 3. Object Oriented Programming
269
3.3 Templates
Immediately after starting OnStart, the function generator is called. It must return a value to
initialize the object AutoPtr in OnStart, and therefore its constructor has not yet been called. Line
02 creates an object AutoPtr#20971 52 inside the function generator and gets a pointer to the first
Dummy#31 45728. Next, a second instance of Dummy#41 94304 is created (line 03), which
replaces the previous copy with descriptor 31 45728 (line 04) in AutoPtr#20971 52, and the old
copy is deleted (line 05). Line 06 creates a temporary AutoPtr#5242880 to return the value from
the generator, and deletes the local one (07). On line 08, the copy constructor for the
AutoPtr#1 048576 object in the function OnStart is finally used, and the pointer from the
temporary object (which is immediately deleted on line 09) is transferred to it. Next, we call Print
with the content of the pointer. When the OnStart completes, the destructor AutoPtr (1 1 )
automatically fires, causing us to also delete the work object Dummy (1 2).
Template technology makes the class AutoPtr a parameterized manager of dynamically allocated
objects. But since AutoPtr has a field T *ptr, it only applies to classes (more precisely, pointers to class
objects). For example, trying to instantiate a template for a string (AutoPtr<string> s) will result in a lot
of errors in the template text, the meaning of which is that the string type does not support pointers.
This is not a problem here, since the purpose of this template is limited to classes, but for more general
templates, this nuance should be kept in mind (see the sidebar).
Pointers and references
Please note that the T * construct cannot appear in templates that you plan to use, including for
built-in types or structures. The point is that pointers in MQL5 are allowed only for classes. This is
not to say that a template cannot in theory be written to apply to both built-in and user-defined
types, but it may require some tweaking. It will probably be necessary either to abandon some of
the functionality or to sacrifice a level of genericity of the template (make several templates
instead of one, overload functions, etc.). 
The most straightforward way to "inject" a pointer type into a template is to include the modifier '*'
along with the actual type when the template is instantiated (i.e. it must match T=Type*). However,
some functions (such as CheckPointer), operators (such as delete), and syntactic constructs (such
as casting ((T)variable)) are sensitive to whether their arguments/operands are pointers or not.
Because of this, the same template text is not always syntactically correct for both pointers and
simple type values.
Another significant type difference to keep in mind: objects are passed to methods by reference
only, but literals (constants) of simple types cannot be passed by reference. Because of this, the
presence or absence of an ampersand may be treated as an error by the compiler, depending on
the inferred type of T. As one of the "workarounds", you can optionally "wrap" argument constants
into objects or variables.
Another trick involves using template methods. We will see it in the next section.
It should be noted that object-oriented techniques go well with patterns. Since a pointer to a base class
can be used to store an object of a derived class, AutoPtr is applicable to objects of any derived Dummy
classes.
In theory, this "hybrid" approach is widely used in the container classes (vector, queue, map, list, etc.),
which, as a rule, are templates. Container classes may, depending on the implementation, impose
additional requirements on the template parameter, in particular, that the inline type must have a copy
constructor and an assignment (copy) operator.

---

## Page 270

Part 3. Object Oriented Programming
270
3.3 Templates
The MQL5 standard library supplied with MetaTrader 5 contains many ready-made templates from this
series: Stack.mqh, Queue.mqh, HashMap.mqh, LinkedList.mqh, RedBlackTree.mqh, and others. They are
all located in the MQL5/Include/Generic directory. True, they do not provide control over dynamic
objects (pointers).
We'll look at our own example of a simple container class in Method templates.
3.3.7 Method templates
Not only an object type as a whole can be a template, but its method separately – simple or static –
also can be a template. The exception is virtual methods: they cannot be made templates. It follows
that template methods cannot be declared inside interfaces. However, interfaces themselves can be
made templates, and virtual methods can be present in class templates.
When a method template is contained within a class/structure template, the parameters of both
templates must be different. If there are multiple template methods, their parameters are not related
in any way and may have the same name.
A method template is declared similar to a function template, but only in the context of a class,
structure, or union (which may or may not be templates).
[ template < typename T ⌠, typename Ti ...] > ]
class class_name
{
   ...
   template < typename U [, typename Ui ...] >
  type method_name(parameters_with_types_T_and_U)
   {
   }
};
Parameters, the return value, and the method body can use types T (general for a class) and U
(specific for a method).
An instance of a method for a specific combination of parameters is generated only when it is called in
the program code.
In the previous section, we described the template class AutoPtr for storing and releasing a single
pointer. When there are many pointers of the same type, it is convenient to put them in a container
object. Let's create a simple template with similar functionality – the class SimpleArray
(SimpleArray.mqh). In order not to duplicate the functionality for controlling the release of dynamic
memory, we will put in the class contract that it is intended for storing values and objects, but not
pointers. To store the pointers, we will place them in AutoPtr objects, and those in the container.
This has another positive effect: because the object AutoPtr is small, it is easy to copy (without
overspending resources on it), which often happens when data is exchanged between functions. The
objects of those application classes that AutoPtr points to can be large, and it is not even necessary to
implement their own copy constructor in them.
Of course, it's cheaper to return pointers from functions, but then you need to reinvent the means of
memory release control. Therefore, it is easier to use a ready-made solution in the form of AutoPtr.
For objects inside the container, we will create the data array of the templated type T.

---

## Page 271

Part 3. Object Oriented Programming
271 
3.3 Templates
template<typename T>
class SimpleArray
{
protected:
   T data[];
   ...
Since one of the main operations for a container is to add an element, let's provide a helper function to
expand the array.
   int expand()
   {
      const int n = ArraySize(data);
      ArrayResize(data, n + 1);
      return n;
   }
We will directly add elements through the overloaded operator '<<'. It uses the generic template
parameter T.
public:
   SimpleArray *operator<<(const T &r)
   {
      data[expand()] = (T)r;
      return &this;
   }
This option takes a value by reference, i.e. a variable or an object. You should pay attention to this for
now, and why this is important will become clear in a couple of moments.
Reading elements is done by overloading the operator '[]' (it has the highest precedence and therefore
does not require the use of parentheses in expressions).
   T operator[](int i) const
   {
      return data[i];
   }
First, let's make sure that the class works on the example of the structure.
struct Properties
{
   int x;
   string s;
};
To do this, we will describe a container for the structure in the function OnStart and place one object
(TemplatesSimpleArray.mq5) into it.

---

## Page 272

Part 3. Object Oriented Programming
272
3.3 Templates
void OnStart()
{
   SimpleArray<Properties> arrayStructs;
   Properties prop = {12345, "abc"};
   arrayStructs << prop;
   Print(arrayStructs[0].x, " ", arrayStructs[0].s);
   ...
}
Debug logging allows you to verify that the structure is in a container.
Now let's try to store some numbers in the container.
   SimpleArray<double> arrayNumbers;
   arrayNumbers << 1.0 << 2.0 << 3.0;
Unfortunately, we will get "parameter passed as reference, variable expected" errors, which occur
exactly in the overloaded operator '<<'.
We need an overload with parameter passing by value. However, we can't just write a similar method
that doesn't have const and '&':
   SimpleArray *operator<<(T r)
   {
      data[expand()] = (T)r;
      return &this;
   }
If you do this, the new variant will lead to an uncompilable template for object types: after all, objects
need to be passed only by reference. Even if the function is not used for objects, it is still present in the
class. Therefore, we will define the new method as a template with its own parameter.
template<typename T>
class SimpleArray
{
   ...
   template<typename U>
   SimpleArray *operator<<(U u)
   {
      data[expand()] = (T)u;
      return &this;
   }
It will appear in the class only if something by value is passed to the operator '<<', which means it is
definitely not an object. True, we cannot guarantee that T and U are the same, so an explicit cast (T)u
is performed. For built-in types (if the two types do not match), in some combinations, conversion with
loss of precision is possible, but the code will compile for sure. The only exception is the prohibition on
converting a string to a boolean type, but it is unlikely that the container will be used for the array bool,
so this restriction is not significant. Those who wish can solve this problem.
With the new template method, the container SimpleArray<double> works as expected and does not
conflict with SimpleArray<Properties> because the two template instances have differences in the
generated source code.

---

## Page 273

Part 3. Object Oriented Programming
273
3.3 Templates
Finally, let's check the container with objects AutoPtr. To do this, let's prepare a simple class Dummy
that will "supply" objects for pointers inside AutoPtr.
class Dummy
{
   int x;
public:
   Dummy(int i) : x(i) { }
   int value() const
   {
      return x;
   }
};
Inside the functionOnStart, let's create a container SimpleArray<AutoPtr<Dummy>> and fill it.
void OnStart()
{
   SimpleArray<AutoPtr<Dummy>> arrayObjects;
   AutoPtr<Dummy> ptr = new Dummy(20);
   arrayObjects << ptr;
   arrayObjects << AutoPtr<Dummy>(new Dummy(30));
   Print(arrayObjects[0][].value());
   Print(arrayObjects[1][].value());
}
Recall that in AutoPtr the operator '[]' is used to return a stored pointer, so arrayObj ects[0][] means:
return the 0th element of the array data into SimpleArray, i.e. the object AutoPtr, and then the second
pair of square brackets is applied to the volume, resulting in a pointer Dummy*. Next, we can work with
all the properties of this object: in this case, we retrieve the current value of the x field.
Because Dummy does not have a copy constructor, you cannot use a container to store these objects
directly without AutoPtr.
   // ERROR:
   // object of 'Dummy' cannot be returned,
   // copy constructor 'Dummy::Dummy(const Dummy &)' not found
   SimpleArray<Dummy> bad;
But a resourceful user can guess how to get around this.
   SimpleArray<Dummy*> bad;
   bad << new Dummy(0);
This code will compile and run. However, this "solution" contains a problem: SimpleArray does not know
how to control pointers, and therefore, when the program exits, a memory leak is detected.
1 undeleted objects left
1 object of type Dummy left
24 bytes of leaked memory
We, as the developers of SimpleArray, have a duty to close this loophole. To do this, let's add another
template method to the class with an overload of the operator '<<' — this time for pointers. Since it is
a template, it is also only included in the resulting source code "on demand": when the programmer
tries to use this overload, that is, write a pointer to the container. Otherwise, the method is ignored.

---

## Page 274

Part 3. Object Oriented Programming
274
3.3 Templates
template<typename T>
class SimpleArray
{
   ...
   template<typename P>
   SimpleArray *operator<<(P *p)
   {
      data[expand()] = (T)*p;
      if(CheckPointer(p) == POINTER_DYNAMIC) delete p;
      return &this;
   }
This specialization throws a compilation error ("object pointer expected") when instantiating a template
with a pointer type. Thus, we inform the user that this mode is not supported.
   SimpleArray<Dummy*> bad; // ERROR is generated in SimpleArray.mqh
In addition, it performs another protective action. If the client class still has a copy constructor, then
saving dynamically allocated objects in the container will no longer lead to a memory leak: a copy of
the object at the passed pointer P *p remains in the container, and the original is deleted. When the
container is destroyed at the end of the OnStart function, its internal array data will automatically call
the destructors for its elements.
void OnStart()
{
   ...
   SimpleArray<Dummy> good;
   good << new Dummy(0);
} // SimpleArray "cleans" its elements
 // no forgotten objects in memory
Method templates and "simple" methods can be defined outside of the main class block (or class
template), similar to what we saw in the Splitting Declaration and Definition of Class section. At the
same time, they are all preceded by the template header (TemplatesExtended.mq5):

---

## Page 275

Part 3. Object Oriented Programming
275
3.3 Templates
template<typename T>
class ClassType
{
   ClassType() // private constructor
   {
      s = &this;
   }
   static ClassType *s; // object pointer (if it was created)
public:
   static ClassType *create() // creation (on first call only)
   {
      static ClassType single; //single pattern for every T
      return single;
   }
   static ClassType *check() // checking pointer without creating
   {
      return s;
   }
   
   template<typename U>
   void method(const U &u);
};
   
template<typename T>
template<typename U>
void ClassType::method(const U &u)
{
   Print(__FUNCSIG__, " ", typename(T), " ", typename(U));
}
   
template<typename T>
static ClassType<T> *ClassType::s = NULL;
It also shows the initialization of a templated static variable, denoting the singleton design pattern.
In the function OnStart, create an instance of the template and test it:
void OnStart()
{
   ClassType<string> *object = ClassType<string>::create();
   double d = 5.0;
   object.method(d);
   // OUTPUT:
   // void ClassType<string>::method<double>(const double&) string double
   
   Print(ClassType<string>::check()); // 1048576 (an example of an instance id)
    Print(ClassType<long>::check());   // 0 (there is no instance for T=long)
}

---

## Page 276

Part 3. Object Oriented Programming
276
3.3 Templates
3.3.8 Nested templates
Templates can be nested within classes/structures or within other class/structure templates. The same
is true for unions.
In the section Unions, we saw the ability to "convert" long 
values   to double and back again without loss
of precision.
Now we can use templates to write a universal "converter"(TemplatesConverter.mq5). The template
class Converter has two parameters T1  and T2, indicating the types between which the conversion will
be performed. To write a value according to the rules of one type and read according to the rules of
another, we again need a union. We will also make it a template (DataOverlay) with parameters U1  and
U2, and define it inside the class.
The class provides a convenient transformation by overloading the operators [], in the implementation
of which the union fields are written and read.
template<typename T1,typename T2>
class Converter
{
private:
   template<typename U1,typename U2>
   union DataOverlay
   {
      U1 L;
      U2 D;
   };
   
   DataOverlay<T1,T2> data;
   
public:
   T2 operator[](const T1 L)
   {
      data.L = L;
      return data.D;
   }
   
   T1 operator[](const T2 D)
   {
      data.D = D;
      return data.L;
   }
};
The union is used to describe the field DataOverlay<T1 ,T2>data within the class. We could use T1  and
T2 directly in DataOverlay and not make this union a template. But to demonstrate the technique itself,
the parameters of the outer template are passed to the inner template when the data field is
generated. Inside the DataOverlay, the same pair of types will be known as U1  and U2 (in addition to
T1  and T2).
Let's see the template in action.

---

## Page 277

Part 3. Object Oriented Programming
277
3.3 Templates
#define MAX_LONG_IN_DOUBLE       9007199254740992
   
void OnStart()
{
   Converter<double,ulong> c;
   
   const ulong value = MAX_LONG_IN_DOUBLE + 1;
   
   double d = value; // possible loss of data due to type conversion
   ulong result = d; // possible loss of data due to type conversion
   
   Print(value == result); // false
   
   double z = c[value];
   ulong restored = c[z];
   
   Print(value == restored); // true
}
3.3.9 Absent template specialization
In some cases, it may be necessary to provide a template implementation for a particular type (or set
of types) in a way that differs from the generic one. For example, it usually makes sense to prepare a
special version of the swap function for pointers or arrays. In such cases, C++ allows you to do what is
called template specialization, that is, to define a version of the template in which the generic type
parameter T is replaced by the required concrete type.
When specializing function and method templates, specific types must be specified for all parameters.
This is called complete specialization.
In the case of C++ object type templates, specialization can be not only complete but also partial: it
specifies the type of only some of the parameters (and the rest will be inferred or specified when the
template is instantiated). There can be several partial specializations: the only condition for this is that
each specialization must describe a unique combination of types.
Unfortunately, there is no specialization in MQL5 in the full sense of the word.
Template function specialization is no different from overloading. For example, given the following
template func:
template<typename T>
void func(T t) { ... }
it is allowed to provide its custom implementation for a given type (such as string) in one of the forms:
// explicit specialization 
template<>
void func(string t) { ... }
or:

---

## Page 278

Part 3. Object Oriented Programming
278
3.3 Templates
// normal overload 
void func(string t) { ... }
Only one of the forms must be selected. Otherwise, we get a compilation error "'func' - function
already defined and has body".
As for the specialization of classes, inheritance from templates with an indication of specific types for
some of the template parameters can be considered as an equivalent of their partial specialization.
Template methods can be overridden in a derived class.
The following example (TemplatesExtended.mq5) shows several options for using template parameters
as parent types, including cases where one of them is specified as specific.

---

## Page 279

Part 3. Object Oriented Programming
279
3.3 Templates
#define RTTI Print(typename(this))
   
class Base
{
public:
   Base() { RTTI; }
};
   
template<typename T> 
class Derived : public T
{
public:
   Derived() { RTTI; }
}; 
   
template<typename T> 
class Base1
{
   Derived<T> object;
public:
   Base1() { RTTI; }
}; 
   
template<typename T>                // complete "specialization"
class Derived1 : public Base1<Base> // 1 of 1 parameter is set 
{
public:
   Derived1() { RTTI; }
}; 
   
template<typename T,typename E> 
class Base2 : public T
{
public:
   Base2() { RTTI; }
}; 
   
template<typename T>                    // partial "specialization"
class Derived2 : public Base2<T,string> // 1 of 2 parameters is set 
{
public:
   Derived2() { RTTI; }
};
We will provide an instantiation of an object according to a template using a variable:
   Derived2<Derived1<Base>> derived2;
Debug type logging using the RTTI macro produces the following result:

---

## Page 280

Part 3. Object Oriented Programming
280
3.3 Templates
Base
Derived<Base>
Base1<Base>
Derived1<Base>
Base2<Derived1<Base>,string>
Derived2<Derived1<Base>>
When developing libraries that come as closed binary, you must ensure that templates are explicitly
instantiated for all types that future users of the library are expected to work with. You can do this by
explicitly calling function templates and creating objects with type parameters in some auxiliary
function, for example, bound to the initialization of a global variable.

---

## Page 281

Part 4. Common APIs
281 
 
Part 4. Common MQL5 APIs
In the previous parts of the book, we studied the basic concepts, syntax, and rules for using MQL5
language constructs. However, this is only a foundation for writing real programs that meet trader
requirements, such as analytical data processing and automatic trading. Solving such tasks would not
be possible without a wide range of built-in functions and means of interaction with the MetaTrader 5
terminal, which make up the MQL5 API.
In this chapter, we will start mastering the MQL5 API and will continue to do so until the end of the
book, gradually getting familiar with all the specialized subsystems.
The list of technologies and capabilities provided to any MQL program by the kernel (the runtime
environment of MQL programs inside the terminal) is very large. This is why it makes sense to start
with the simplest things that can be useful in most programs. In particular, here we will look at
functions specialized for work with arrays, strings, files, data transformation, user interaction,
mathematical functions, and environmental control.
Previously, we learned to describe our own functions in MQL5 and call them. The built-in functions of
the MQL5 API are available from the source code, as they say, "out of the box", i.e. without any
preliminary description.
It is important to note that, unlike in C++, no additional preprocessor directives are required to include
a specific set of built-in functions in a program. The names of all MQL5 API functions are present in the
global context (namespace), always and unconditionally.
On the one hand, this is convenient, but on the other hand, it requires you to be aware of a possible
name conflict. If you accidentally try to use one of the names of the built-in functions, it will override
the standard implementation, which can lead to unexpected consequences: at best, you get a compiler
error about ambiguous overload, and at worst, all the usual calls will be redirected to the “new”
implementation, without any warnings.
In theory, similar names can be used in other contexts, for example, as a class method name or in a
dedicated (user) namespace. In such cases, calling a global function can be done using the context
resolution operator: we discussed this situation in the section Nested types, namespaces, and the '::'
context operator.
MQL5 Programming for Traders – Source Codes from the Book. Part 4
Examples from the book are also available in the public project \MQL5\Shared Projects\MQL5Book
4.1  Built-in type conversions
Programs often operate with different data types. We have already encountered mechanisms of explicit
and implicit casting of built-in types in the Types Casting section. They provide universal conversion
methods that are not always suitable, for one reason or another. The MQL5 API provides a set of
conversion functions using which a programmer can manage data conversions from one type to another
and configure conversion results.
Among the most frequently used functions are those which convert various types to strings or vice
versa. Specifically, this includes conversions for numbers, dates and times, colors, structures, and
enums. Some types have additional specific operations.

---

## Page 282

Part 4. Common APIs
282
4.1  Built-in type conversions
This section considers various data conversion methods, providing programmers with the necessary
tools to work with a variety of data types in trading robots. It includes the following subsections:
Numbers to strings and vice versa:
·This subsection explores methods for converting numerical values to strings and vice versa. It
covers important aspects such as number formatting and handling various number systems.
Normalization of doubles:
·Normalizing double numbers is an important aspect when working with financial data. This section
discusses normalization methods, ways to avoid precision loss, and processing floating-point values.
Date and time:
· Conversion of date and time plays a key role in trading strategies. This subsection discusses
methods for working with dates, time intervals, and special data types like datetime.
Color:
·In MQL5, colors are represented by a special data type. The subsection examines the conversion of
color values, their representation and use in graphical elements of trading robots.
Structures:
·Data conversion within structures is an important topic when dealing with complex structured data.
We will see methods of interacting with structures and their elements.
Enumerations:
·Enumerations provide named constants and enhance code readability. This subsection discusses
how to convert enumeration values and effectively use them in a program.
Type complex:
·The complex type is designed to work with complex numbers. This section considers methods for
converting and using complex numbers.
We will study all such functions in this chapter.
4.1 .1  Numbers to strings and vice versa
Numbers to strings and back, strings to numbers, can be converted using the explicit type casting
operator. For example, for types double and string, it might look like this:
double number = (double)text;
string text = (string)number;
Strings can be converted to other numeric types, such as float, long, int, etc.
Note that casting to a real type (float) provides fewer significant digits, which in some applications may
be considered an advantage as it gives a more compact and easier-to-read representation.
Strictly speaking, this type casting is not mandatory, since even if there is no explicit cast operator,
the compiler will produce type casting implicitly. However, you will receive a compiler warning in this
case, and thus it is recommended to always make type castings explicit.

---

## Page 283

Part 4. Common APIs
283
4.1  Built-in type conversions
The MQL5 API provides some other useful functions, which are described below. The descriptions are
followed by a general example.
double StringToDouble(string text)
The StringToDouble function converts a string to a double number.
It is a complete analog of type casting to (double). Its practical purpose is actually limited to
preserving backward compatibility with legacy source codes. The preferred method is type casting, as
it is more compact and is implemented within the syntax of the language.
According to the conversion process, a string should contain a sequence of characters that meet the
rules for writing literals of numeric types (both float and integer). In particular, a string may begin with
a '+' or '-' sign, followed by a digit, and may continue further as a sequence of digits.
Real numbers can contain a single dot character '.' separating the fractional part and an optional
exponent in the following format: character 'e' or 'E' followed by a sequence of digits for the degree (it
can also be preceded by a '+' or '-').
For integers, hexadecimal notation is supported, i.e., the "0x" prefix can be followed not only by
decimal digits but also by 'A', 'B', 'C', 'D', 'E', 'F' (in any position).
When any non-expected character (such as a letter, punctuation mark, second period, or intermediate
space) is encountered in the string, the conversion ends. In this case, if there were allowed characters
before this position, they are interpreted as a number, and if not, the result will be 0.
Initial empty characters (spaces, tabs, newlines) are skipped and do not affect the conversion. If they
are followed by numbers and other characters that meet the rules, the number will be received
correctly.
The following table provides some examples of valid conversions with explanations.
string
double
Result
"1 23.45"
1 23.45
One decimal point
"\t   1 23"
1 23.0
Whitespace characters at the beginning are ignored
"-1 2345"
-1 2345.0
A signed number
"1 23e-5"
0.001 23
Scientific notation with exponent
"0x1 2345"
74565.0
Hexadecimal notation
The following table shows examples of incorrect conversions.
string
double
Result
"x1 2345"
0.0
Starts with an unresolved character (letter)
"1 23x45"
1 23.0
The letter after 1 23 breaks conversion
"   1 2 3"
1 2.0
The space after 1 2 breaks the conversion

---

## Page 284

Part 4. Common APIs
284
4.1  Built-in type conversions
string
double
Result
"1 23.4.5"
1 23.4
The second decimal point after 1 23.4 breaks the conversion
"1 ,234.50"
1 .0
The comma after 1  breaks conversion
"-+1 2345"
0.0
Too many signs (two)
string DoubleToString(double number, int digits = 8)
The DoubleToString function converts a number to a string with the specified precision (number of
digits from -1 6 to 1 6).
It does a job similar to casting a number to (string) but allows you to choose, using the second
parameter, the number precision in the resulting string.
The operator (string) applied to double, displays 1 6 significant digits (total, including mantissa and
fractional part). The full equivalent of this cannot be achieved with a function.
If the digits parameter is greater than or equal to 0, it indicates the number of decimal places. In this
case, the number of characters before the decimal mark is determined by the number itself (how large
it is), and if the total number of characters in the mantissa and that indicated in digits turns out to be
greater than 1 6, then the least significant digits will contain "garbage" (due to how the real numbers
are stored). 1 6 characters represent the average maximum precision for type double, i.e., setting digits
to 1 6 (maximum) will only provide an accurate representation of values less than 1 0.
If the digits parameter is less than 0, it specifies the number of significant digits, and this number will
be output in scientific format with an exponent. In terms of precision (but not recording format),
setting digits=-1 6 in the function generates a result close to casting (string).
The function, as a rule, is used for uniform formatting of data sets (including right-alignment of a
column of a certain table), in which values have the same decimal precision (for example, the number
of decimal places in the financial instrument price or a lot size).
Please note that errors may occur during mathematical calculations, causing the result to be not a
valid number although it has the type double (or float). For example, a variable might contain the
result of calculating the square root of a negative number.
Such values are called "Not a Number" (NaN) and are displayed when cast to (string) as a short
hint of error type, for example, -nan(ind) (ind - undefined), nan(inf) (inf - infinity). When using the
DoubleToString function, you will get a large number that makes no sense.
It is especially important that all subsequent calculations with NaN will also give NaN. To check
such values, there is the MathIsValidNumber function.
long StringToInteger(string text)
The function converts a string to a number of type long. Note that the result type is definitely long, and
not int (despite the name) and not ulong.
An alternative way is to typecast using the operator (long). Moreover, any other integer type of your
choice can be used for the cast:(int), (uint), (ulong), etc.

---

## Page 285

Part 4. Common APIs
285
4.1  Built-in type conversions
The conversion rules are similar to the type double, but exclude the dot character and the exponent
from the allowed characters.
string IntegerToString(long number, int length = 0, ushort filling = ' ')
Function IntegerToString converts an integer of type long to a string of the specified length. If the
number representation takes less than one character, it is left-padded with a character filling (with a
space by default). Otherwise, the number is displayed in its entirety, without restriction. Calling a
function with default parameters is equivalent to casting to (string).
Of course, smaller integer types (for example, int, short) will be processed by the function without
problems.
Examples of using all the above functions are given in the script ConversionNumbers.mq5.

---

## Page 286

Part 4. Common APIs
286
4.1  Built-in type conversions
void OnStart()
{
   const string text = "123.4567890123456789";
   const string message = "-123e-5 buckazoid";
   const double number = 123.4567890123456789;
   const double exponent = 1.234567890123456789e-5;
   
   // type casting
   Print((double)text);    // 123.4567890123457
   Print((double)message); // -0.00123
   Print((string)number);  // 123.4567890123457
   Print((string)exponent);// 1.234567890123457e-05
   Print((long)text);      // 123
   Print((long)message);   // -123
   
   // converting with functions
   Print(StringToDouble(text)); // 123.4567890123457
   Print(StringToDouble(message)); // -0.00123
   
   // by default, 8 decimal digits
   Print(DoubleToString(number)); // 123.45678901
   
   // custom precision
   Print(DoubleToString(number, 5));  // 123.45679
   Print(DoubleToString(number, -5)); // 1.23457e+02
   Print(DoubleToString(number, -16));// 1.2345678901234568e+02
   Print(DoubleToString(number, 16)); // 123.4567890123456807
   // last 2 digits are not accurate!
   Print(MathSqrt(-1.0));                 // -nan(ind)
   Print(DoubleToString(MathSqrt(-1.0))); // 9223372129088496176.54775808
   
   Print(StringToInteger(text));      // 123
   Print(StringToInteger(message));   // -123
   
   Print(IntegerToString(INT_MAX));         // '2147483647'
   Print(IntegerToString(INT_MAX, 5));      // '2147483647'
   Print(IntegerToString(INT_MAX, 16));     // '      2147483647'
   Print(IntegerToString(INT_MAX, 16, '0'));// '0000002147483647'
}
4.1 .2 Normalization of doubles
The MQL5 API provides a function for rounding floating point numbers to a specified precision (the
number of significant digits in the fractional part).
double NormalizeDouble(double number, int digits)
Rounding is required in trading algorithms to set volumes and prices in orders. Rounding is performed
according to the standard rules: the last visible digit is increased by 1  if the next (discarded) digit is
greater than or equal to 5.
Valid values of the parameter digits: 0 to 8.

---

## Page 287

Part 4. Common APIs
287
4.1  Built-in type conversions
Examples of using the function are available in the ConversionNormal.mq5 file.
void OnStart()
{
   Print(M_PI);                      // 3.141592653589793
   Print(NormalizeDouble(M_PI, 16)); // 3.14159265359
   Print(NormalizeDouble(M_PI, 8));  // 3.14159265
   Print(NormalizeDouble(M_PI, 5));  // 3.14159
   Print(NormalizeDouble(M_PI, 1));  // 3.1
   Print(NormalizeDouble(M_PI, -1)); // 3.14159265359
   ...
Due to the fact that any real number has a limited internal representation precision, the number can be
displayed approximately even when normalized:
   ...
   Print(512.06);                    // 512.0599999999999
   Print(NormalizeDouble(512.06, 5));// 512.0599999999999
   Print(DoubleToString(512.06, 5)); // 512.06000000
   Print((float)512.06);             // 512.06
}
This is normal and inevitable. For more compact formatting, use the functions DoubleToString,
StringFormat or intermediate casting to (float).
To round a number up or down to the nearest integer, use the functions MathRound, MathCeil,
MathFloor (see section Rounding functions).
4.1 .3 Date and Time
Values of type datetime intended for storing date and/or time usually undergo several types of
conversion:
• into lines and back to display data to the user and to read data from external sources
• into special structures MqlDateTime (see below) to work with individual date and time components
• to the number of seconds elapsed since 01 /01 /1 970, which corresponds to the internal
representation of datetime and is equivalent to the integer type long
For the last item, use datetime to (long) casting, or vice versa, long To (datetime), but note that the
supported date range is from January 1 , 1 970 (value 0) to December 31 , 3000 (3253521 5999
seconds).
For the first two options, the MQL5 API provides the following functions.
string TimeToString(datetime value, int mode = TIME_DATE |  TIME_MINUTES)
Function TimeToString converts a value of type datetime to a string with date and time components,
according to the mode parameter in which you can set an arbitrary combination of flags:
• TIME_DATE – date in the format "YYYY.MM.DD"
• TIME_MINUTES – time in the format "hh:mm", i.e., with hours and minutes
• TIME_SECONDS – time in "hh:mm:ss" format, i.e. with hours, minutes and seconds

---

## Page 288

Part 4. Common APIs
288
4.1  Built-in type conversions
To output the date and time data in full, you can set mode equal to TIME_DATE |  TIME_SECONDS (the
TIME_DATE |  TIME_MINUTES |  TIME_SECONDS option will also work, but is redundant). This is
equivalent to casting a value of type datetime to (string).
Usage examples are provided in the ConversionTime.mq5 file.
#define PRT(A) Print(#A, "=", (A))
void OnStart()
{
   datetime time = D'2021.01.21 23:00:15';
   PRT((string)time);
   PRT(TimeToString(time));
   PRT(TimeToString(time, TIME_DATE | TIME_MINUTES | TIME_SECONDS));
   PRT(TimeToString(time, TIME_MINUTES | TIME_SECONDS));
   PRT(TimeToString(time, TIME_DATE | TIME_SECONDS));
   PRT(TimeToString(time, TIME_DATE));
   PRT(TimeToString(time, TIME_MINUTES));
   PRT(TimeToString(time, TIME_SECONDS));
}
The script will print the following log:
(string)time=2021.01.21 23:00:15
TimeToString(time)=2021.01.21 23:00
TimeToString(time,TIME_DATE|TIME_MINUTES|TIME_SECONDS)=2021.01.21 23:00:15
TimeToString(time,TIME_MINUTES|TIME_SECONDS)=23:00:15
TimeToString(time,TIME_DATE|TIME_SECONDS)=2021.01.21 23:00:15
TimeToString(time,TIME_DATE)=2021.01.21
TimeToString(time,TIME_MINUTES)=23:00
TimeToString(time,TIME_SECONDS)=23:00:15
datetime StringToTime(string value)
The function StringToTime converts a string containing a date and/or time to a value of type datetime.
The string can contain only the date, only the time, or the date and time together.
The following formats are recognized for dates:
• "YYYY.MM.DD"
• "YYYYMMDD"
• "YYYY/MM/DD"
• "YYYY-MM-DD"
• "DD.MM.YYYY"
• "DD/MM/YYYY"
• "DD-MM-YYYY"
The following formats are supported for time:
• "hh:mm"
• "hh:mm:ss"

---

## Page 289

Part 4. Common APIs
289
4.1  Built-in type conversions
• "hhmmss"
There must be at least one space between the date and time.
If only time is present in the string, the current date will be substituted in the result. If only the date is
present in the string, the time will be set to 00:00:00.
If the supported syntax in the string is broken, the result is the current date.
The function usage examples are given in the script ConversionTime.mq5.
void OnStart()
{
   string timeonly = "21:01";   // time only
   PRT(timeonly);
   PRT((datetime)timeonly);
   PRT(StringToTime(timeonly));
   
   string date = "2000-10-10";  // date only
   PRT((datetime)date);
   PRT(StringToTime(date));
   PRT((long)(datetime)date);
   long seconds = 60;
   PRT((datetime)seconds); // 1 minute from the beginning of 1970
   
   string ddmmyy = "15/01/2012 01:02:03"; // date and time, and the date in
   PRT(StringToTime(ddmmyy));             // in "forward" order, still ok
   
   string wrong = "January 2-nd";
   PRT(StringToTime(wrong));
}
In the log, we will see something like the following (####.##.## is the current date the script was
launched):
timeonly=21:01
(datetime)timeonly=####.##.## 21:01:00
StringToTime(timeonly)=####.##.## 21:01:00
(datetime)date=2000.10.10 00:00:00
StringToTime(date)=2000.10.10 00:00:00
(long)(datetime)date=971136000
(datetime)seconds=1970.01.01 00:01:00
StringToTime(ddmmyy)=2012.01.15 01:02:03
(datetime)wrong=####.##.## 00:00:00
In addition to StringToTime, you can use the cast operator (datetime) to convert strings to dates and
times. However, the advantage of the function is that when an incorrect source string is detected, the
function sets an internal variable with an error code _ LastError (which is also available via the function
GetLastError). Depending on which part of the string contains uninterpreted data, the error code could
be ERR_WRONG_STRING_DATE (5031 ), ERR_WRONG_STRING_TIME (5032) or another option from
the list related to getting the date and time from the string.

---

## Page 290

Part 4. Common APIs
290
4.1  Built-in type conversions
bool TimeToStruct(datetime value, MqlDateTime &struct)
To parse date and time components separately, the MQL5 API provides the TimeToStruct function
which converts a value of type datetime into the MqlDateTime structure:
struct MqlDateTime
{ 
   int year;           // year
   int mon;            // month
   int day;            // day
   int hour;           // hour
   int min;            // minutes
   int sec;            // seconds
   int day_of_week;    // day of the week
   int day_of_year;    // the number of the day in a year (January 1 has number 0)
};
The days of the week are numbered in the American manner: 0 for Sunday, 1  for Monday, and so on up
to 6 for Saturday. They can be identified using the built-in ENUM_DAY_OF_WEEK enumeration.
The function returns true if successful and false on error, in particular, if an incorrect date is passed.
Let's check the performance of the function using the ConversionTimeStruct.mq5 script. To do this,
let's create the time array of type datetime with test values. We will call TimeToStruct for each of them
in a loop.
The results will be added to an array of structures MqlDateTime mdt[]. We will first initialize it with
zeros, but since the built-in function ArrayInitialize does not know how to handle structures, we will
have to write an overload for it (in the future we will learn an easier way to fill an array with zeros: in
the section Zeroing objects and arrays the function ZeroMemory will be introduced).
int ArrayInitialize(MqlDateTime &mdt[], MqlDateTime &init)
{
   const int n = ArraySize(mdt);
   for(int i = 0; i < n; ++i)
   {
      mdt[i] = init;
   }
   return n;
}
After the process, we will output the array of structures to the log using the built-in function ArrayPrint.
This is the easiest way to provide nice data formatting (it can be used even if there is only one
structure: just put it in an array of size 1 ).

---

## Page 291

Part 4. Common APIs
291 
4.1  Built-in type conversions
void OnStart()
{
   // fill the array with tests
   datetime time[] =
   {
      D'2021.01.28 23:00:15', // valid datetime value
      D'3000.12.31 23:59:59', // the largest supported date and time
      LONG_MAX // invalid date: will cause an error ERR_INVALID_DATETIME (4010)
   };
   
   // calculate the size of the array at compile time
   const int n = sizeof(time) / sizeof(datetime);
   
   MqlDateTime null = {}; // example with zeros
   MqlDateTime mdt[];
   
   // allocating memory for an array of structures with results
   ArrayResize(mdt, n);
   
   // call our ArrayInitialize overload 
   ArrayInitialize(mdt, null);
   
   // run tests
   for(int i = 0; i < n; ++i)
   {
      PRT(time[i]); // displaying initial data
   
      if(!TimeToStruct(time[i], mdt[i])) // if an error occurs, output its code
      {
         Print("error: ", _LastError);
         mdt[i].year = _LastError;
      }
   }
   
   // output the results to the log
   ArrayPrint(mdt);
   ...
}
As a result, we get the following strings in the log:

---

## Page 292

Part 4. Common APIs
292
4.1  Built-in type conversions
time[i]=2021.01.28 23:00:15
time[i]=3000.12.31 23:59:59
time[i]=wrong datetime
wrong datetime -> 4010
    [year] [mon] [day] [hour] [min] [sec] [day_of_week] [day_of_year]
[0]   2021     1    28     23     0    15             4            27
[1]   3000    12    31     23    59    59             3           364
[2]   4010     0     0      0     0     0             0             0
You can make sure that all fields have received the appropriate values. For incorrect initial dates, we
store the error code in the year field (in this case, there is only one such error: 401 0,
ERR_INVALID_DATETIME).
Recall that for the maximum date value in MQL5, the DATETIME_MAX constant is introduced, equal to
the integer value 0x793406fff, which corresponds to 23:59:59 December 31 , 3000.
The most common problem that is solved using the function TimeToStruct, is getting the value of a
particular date/time component. Therefore, it makes sense to prepare an auxiliary header file
(MQL5Book/DateTime.mqh) with a ready implementation option. The file has the datetime class.

---

## Page 293

Part 4. Common APIs
293
4.1  Built-in type conversions
class DateTime
{
private:
   MqlDateTime mdtstruct;
   datetime origin;
   
   DateTime() : origin(0)
   {
      TimeToStruct(0, mdtstruct);
   }
   
   void convert(const datetime &dt)
   {
      if(origin != dt)
      {
         origin = dt;
         TimeToStruct(dt, mdtstruct);
      }
   }
   
public:
   static DateTime *assign(const datetime dt)
   {
      _DateTime.convert(dt);
      return &_DateTime;
   }
   ENUM_DAY_OF_WEEK timeDayOfWeek() const
   {
      return (ENUM_DAY_OF_WEEK)mdtstruct.day_of_week;
   }
   int timeDayOfYear() const
   {
      return mdtstruct.day_of_year;
   }
   int timeYear() const
   {
      return mdtstruct.year;
   }
   int timeMonth() const
   {
      return mdtstruct.mon;
   }
   int timeDay() const
   {
      return mdtstruct.day;
   }
   int timeHour() const
   {
      return mdtstruct.hour;
   }
   int timeMinute() const

---

## Page 294

Part 4. Common APIs
294
4.1  Built-in type conversions
   {
      return mdtstruct.min;
   }
   int timeSeconds() const
   {
      return mdtstruct.sec;
   }
   
   static DateTime _DateTime;
};
   
static DateTime DateTime::_DateTime;
The class comes with several macros that make it easier to call its methods.
#define TimeDayOfWeek(T) DateTime::assign(T).timeDayOfWeek()
#define TimeDayOfYear(T) DateTime::assign(T).timeDayOfYear()
#define TimeYear(T) DateTime::assign(T).timeYear()
#define TimeMonth(T) DateTime::assign(T).timeMonth()
#define TimeDay(T) DateTime::assign(T).timeDay()
#define TimeHour(T) DateTime::assign(T).timeHour()
#define TimeMinute(T) DateTime::assign(T).timeMinute()
#define TimeSeconds(T) DateTime::assign(T).timeSeconds()
   
#define _TimeDayOfWeek DateTime::_DateTime.timeDayOfWeek
#define _TimeDayOfYear DateTime::_DateTime.timeDayOfYear
#define _TimeYear DateTime::_DateTime.timeYear
#define _TimeMonth DateTime::_DateTime.timeMonth
#define _TimeDay DateTime::_DateTime.timeDay
#define _TimeHour DateTime::_DateTime.timeHour
#define _TimeMinute DateTime::_DateTime.timeMinute
#define _TimeSeconds DateTime::_DateTime.timeSeconds
The class has the mdtstruct field of the MqlDateTime structure type. This field is used in all internal
conversions. Structure fields are read using getter methods: a corresponding method is allocated for
each field.
One static instance is defined inside the class: _ DateTime (one object is enough, because all MQL
programs are single-threaded). The constructor is private, so trying to create other datetime objects
will fail.
Using macros, we can conveniently receive separate components from datetime, for example, the year
(TimeYear(T)), month (TimeMonth(T)), number (TimeDay(T)), or day of the week (TimeDayOfWeek(T)).
If from one value of datetime it is necessary to receive several fields, then it is better to use similar
macros in all calls except the first one without a parameter and starting with the underscore symbol:
they read the desired field from the structure without re-setting the date/time and calling the
TimeToStruct function. For example:

---

## Page 295

Part 4. Common APIs
295
4.1  Built-in type conversions
   // use the DateTime class from MQL5Book/DateTime.mqh:
   // first get the day of the week for the specified datetime value
   PRT(EnumToString(TimeDayOfWeek(time[0])));
   // then read year, month and day for the same value
   PRT(_TimeYear());
   PRT(_TimeMonth());
   PRT(_TimeDay());
The following strings should appear in the log.
EnumToString(DateTime::_DateTime.assign(time[0]).__TimeDayOfWeek())=THURSDAY
DateTime::_DateTime.__TimeYear()=2021
DateTime::_DateTime.__TimeMonth()=1
DateTime::_DateTime.__TimeDay()=28
The built-in function EnumToString converts an element of any enumeration into a string. It will be
described in a separate section.
datetime StructToTime(MqlDateTime &struct)
The StructToTime function performs a conversion from the MqlDateTime structure (see above the
description of the TimeToStruct function) containing date and time components, into a value of type
datetime. The fields day_ of_ week  and day_ of_ year are not used.
If the state of the remaining fields is invalid (corresponding to a non-existent or unsupported date), the
function may return either a corrected value, or WRONG_VALUE (-1  in the representation of type long),
depending on the problem. Therefore, you should check for an error based on the state of the global
variable _ LastError. A successful conversion is completed with code 0. Before converting, you should
reset a possible failed state in _ LastError (preserved as an artifact of the execution of some previous
instructions) using the ResetLastError function.
The StructToTime function test is also provided in the script ConversionTimeStruct.mq5. The array of
structures parts is converted to datetime in the loop.

---

## Page 296

Part 4. Common APIs
296
4.1  Built-in type conversions
   MqlDateTime parts[] =
   {
      {0, 0, 0, 0, 0, 0, 0, 0},
      {100, 0, 0, 0, 0, 0, 0, 0},
      {2021, 2, 30, 0, 0, 0, 0, 0},
      {2021, 13, -5, 0, 0, 0, 0, 0},
      {2021, 50, 100, 0, 0, 0, 0, 0},
      {2021, 10, 20, 15, 30, 155, 0, 0},
      {2021, 10, 20, 15, 30, 55, 0, 0},
   };
   ArrayPrint(parts);
   Print("");
   
   // convert all elements in the loop
   for(int i = 0; i < sizeof(parts) / sizeof(MqlDateTime); ++i)
   {
      ResetLastError();
      datetime result = StructToTime(parts[i]);
      Print("[", i, "] ", (long)result, " ", result, " ", _LastError);
   }
For each element, the resulting value and an error code are displayed.
       [year] [mon] [day] [hour] [min] [sec] [day_of_week] [day_of_year]
   [0]      0     0     0      0     0     0             0             0
   [1]    100     0     0      0     0     0             0             0
   [2]   2021     2    30      0     0     0             0             0
   [3]   2021    13    -5      0     0     0             0             0
   [4]   2021    50   100      0     0     0             0             0
   [5]   2021    10    20     15    30   155             0             0
   [6]   2021    10    20     15    30    55             0             0
   
   [0] -1 wrong datetime 4010
   [1] 946684800 2000.01.01 00:00:00 4010
   [2] 1614643200 2021.03.02 00:00:00 0
   [3] 1638316800 2021.12.01 00:00:00 4010
   [4] 1640908800 2021.12.31 00:00:00 4010
   [5] 1634743859 2021.10.20 15:30:59 4010
   [6] 1634743855 2021.10.20 15:30:55 0
Note that the function corrects some values without raising the error flag. So, in element number 2, we
passed the date, February 30, 2021 , into the function, which was converted to March 2, 2021 , and
_ LastError = 0.
4.1 .4 Color
The MQL5 API contains 3 built-in functions to work with the color: two of them serve for conversion of
type color to and from a string, and the third one provides a special color representation with
transparency (ARGB).

---

## Page 297

Part 4. Common APIs
297
4.1  Built-in type conversions
string ColorToString(color value, bool showName = false)
The ColorToString function converts the passed color value to a string like "R,G,B" (where R, G, B are
numbers from 0 to 255, corresponding to the intensity of the red, green, and blue component in the
color) or to the color name from the list of predefined web colors if the showName parameter equals
true. The color name is only returned if the color value exactly matches one of the webset.
Examples of using the function are given in the ConversionColor.mq5  script.
void OnStart()
{
   Print(ColorToString(clrBlue));            // 0,0,255
   Print(ColorToString(C'0, 0, 255', true)); // clrBlue
   Print(ColorToString(C'0, 0, 250'));       // 0,0,250
   Print(ColorToString(C'0, 0, 250', true)); // 0,0,250 (no name for this color)
   Print(ColorToString(0x34AB6821, true));   // 33,104,171 (0x21,0x68,0xAB)
}
color StringToColor(string text)
The StringToColor function converts a string like "R,G,B" or a string containing the name of a standard
web color into a value of type color. If the string does not contain a properly formatted triplet of
numbers or a color name, the function will return 0 (clrBlack).
Examples can be seen in the script ConversionColor.mq5.
void OnStart()
{
   Print(StringToColor("0,0,255")); // clrBlue
   Print(StringToColor("clrBlue")); // clrBlue
   Print(StringToColor("Blue"));    // clrBlack (no color with that name)
   // extra text will be ignored
   Print(StringToColor("255,255,255 more text"));      // clrWhite
   Print(StringToColor("This is color: 128,128,128")); // clrGray
}
uint ColorToARGB(color value, uchar alpha = 255)
The ColorToARGB function converts a value of type color and one-byte value alpha (specifying
transparency) into an ARGB representation of a color (a value of type uint). The ARGB color format is
used when creating graphic resources and text drawing on charts.
The alpha value can vary from 0 to 255. "0" corresponds to full color transparency (when displaying a
pixel of this color, it leaves the existing graph image at this point unchanged), 255 means applying full
color density (when displaying a pixel of this color, it completely replaces the color of the graph at the
corresponding point). The value 1 28 (0x80) is translucent.
As we know the type color describes a color using three color components: red (Red), green
(Green) and blue (Blue), which are stored in the format 0x00BBGGRR in a 4-byte integer (uint).
Each component is a byte that specifies the saturation of that color in the range 0 to 255 (0x00 to
0xFF in hexadecimal). The highest byte is empty. For example, white color contains all colors and
therefore has a meaning color equal to 0xFFFFFF.

---

## Page 298

Part 4. Common APIs
298
4.1  Built-in type conversions
But in certain tasks, it is required to specify the color transparency in order to describe how the
image will look when superimposed on some background (on another, already existing image). For
such cases, the concept of an alpha channel is introduced, which is encoded by an additional byte.
The ARGB color representation, together with the alpha channel (denoted AA), is 0xAARRGGBB. For
example, the value 0x80FFFF00 means yellow (a mix of the red and green components) translucent
color.
When overlaying an image with an alpha channel on some background, the resulting color is obtained:
Cresult = (Cforeground * alpha + Cbackground * (255 - alpha)) / 255
where C takes the value of each of the R, G, B components, respectively. This formula is provided for
reference. When using built-in functions with ARGB colors, transparency is applied automatically.
An example of ColorToARGB application is given in ConversionColor.mq5. An auxiliary structure Argb and
union ColorARGB have been added to the script for convenience when analyzing color components.
struct Argb
{
   uchar BB;
   uchar GG;
   uchar RR;
   uchar AA;
};
   
union ColorARGB
{
   uint value;
   uchar channels[4]; // 0 - BB, 1 - GG, 2 - RR, 3 - AA
   Argb split[1];
   ColorARGB(uint u) : value(u) { }
};
The structure is used as the split-type field in the union and provides access to the ARGB components
by name. The union also has a byte array channels, which allows you to access components by index.

---

## Page 299

Part 4. Common APIs
299
4.1  Built-in type conversions
void OnStart()
{
   uint u = ColorToARGB(clrBlue);
   PrintFormat("ARGB1=%X", u); // ARGB1=FF0000FF
   ColorARGB clr1(u);
   ArrayPrint(clr1.split);
   /*
       [BB] [GG] [RR] [AA]
   [0]  255    0    0  255
   */
   
   u = ColorToARGB(clrDeepSkyBlue, 0x40);
   PrintFormat("ARGB2=%X", u); // ARGB2=4000BFFF
   ColorARGB clr2(u);
   ArrayPrint(clr2.split);
   /*
       [BB] [GG] [RR] [AA]
   [0]  255  191    0   64
   */
}
We will consider the print format function a little later, in the corresponding section.
There is no built-in function to convert ARGB back to color (because it is not usually required), but
those who wish to do so, can use the following macro:
#define ARGBToColor(U) (color) \
   ((((U) & 0xFF) << 16) | ((U) & 0xFF00) | (((U) >> 16) & 0xFF))
4.1 .5 Structures
When integrating MQL programs with external systems, in particular, when sending or receiving data via
the Internet, it becomes necessary to convert data structures into byte arrays. For these purposes,
the MQL5 API provides two functions: StructToCharArray and CharArrayToStruct.
In both cases, it is assumed that a structure contains only simple built-in types, that is, all built-in
types except lines and dynamic arrays. A structure can also contain other simple structures. Class
objects and pointers are not allowed. Such structures are also called POD (Plain Old Data).
bool StructToCharArray(const void &object, uchar &array[], uint pos = 0)
The StructToCharArray function copies the POD structure obj ect into the array array of type uchar.
Optionally, using the parameter pos you can specify the position in the array, starting from which the
bytes will be placed. By default, copying goes to the beginning of the array, and the dynamic array will
be automatically increased in size if its current size is not enough for the entire structure.
The function returns a success indicator (true) or errors (false).
Let's check its performance with the script ConversionStruct.mq5. Let's create a new structure type
DateTimeMsc, which includes the standard structure MqlDateTime (field mdt) and an additional field
msc of type int to store milliseconds.

---

## Page 300

Part 4. Common APIs
300
4.1  Built-in type conversions
struct DateTimeMsc
{
   MqlDateTime mdt;
   int msc;
   DateTimeMsc(MqlDateTime &init, int m = 0) : msc(m)
   {
      mdt = init;
   }
};
Inside the OnStart function, let's convert a test value datetime to our structure, and then to the byte
array.
MqlDateTime TimeToStructInplace(datetime dt)
{
   static MqlDateTime m;
   if(!TimeToStruct(dt, m))
   {
      // the error code, _LastError, can be displayed
      // but here we just return zero time
      static MqlDateTime z = {};
      return z;
   }
   return m;
}
#define MDT(T) TimeToStructInplace(T)
void OnStart()
{
   DateTimeMsc test(MDT(D'2021.01.01 10:10:15'), 123);
   uchar a[];
   Print(StructToCharArray(test, a));
   Print(ArraySize(a));
   ArrayPrint(a);
}
We will get the following result in the log (the array is reformatted with additional line breaks to
emphasize the correspondence of bytes to each of the fields):
   true
   36
   229   7   0   0
     1   0   0   0
     1   0   0   0
    10   0   0   0
    10   0   0   0
    15   0   0   0
     5   0   0   0
     0   0   0   0
   123   0   0   0

---

## Page 301

Part 4. Common APIs
301 
4.1  Built-in type conversions
bool CharArrayToStruct(void &object, const uchar &array[], uint pos = 0)
The CharArrayToStruct function copies the array array of the uchar type to the POD structure obj ect.
Using the pos parameter, you can specify the position in the array from which to start reading bytes.
The function returns a success indicator (true) or errors (false).
Continuing the same example (ConversionStruct.mq5), we can restore the original date and time from
the byte array.
void OnStart()
{
   ...
   DateTimeMsc receiver;
   Print(CharArrayToStruct(receiver, a));                 // true
   Print(StructToTime(receiver.mdt), "'", receiver.msc);  // 2021.01.01 10:10:15'123
}
4.1 .6 Enumerations
In MQL5 API, an enumeration value can be converted to a string using the EnumToString function.
There is no ready-made inverse transformation.
string EnumToString(enum value)
The function converts the value (i.e., the ID of the passed element) of an enumeration of any type to a
string.
Let's use it to solve one of the most popular tasks: to find out the size of the enumeration (how many
elements it contains) and exactly what values correspond to all elements. For this purpose, in the
header file EnumToArray.mqh we implement the special template function (due to the template type E,
it will work for any enum):
template<typename E>
int EnumToArray(E dummy, int &values[],
   const int start = INT_MIN, 
   const int stop = INT_MAX)
{
   const static string t = "::";
   
   ArrayResize(values, 0);
   int count = 0;
   
   for(int i = start; i < stop && !IsStopped(); i++)
   {
      E e = (E)i;
      if(StringFind(EnumToString(e), t) == -1)
      {
         ArrayResize(values, count + 1);
         values[count++] = i;
      }
   }
   return count;

---

## Page 302

Part 4. Common APIs
302
4.1  Built-in type conversions
}
The concept of its operation is based on the following. Since enumerations in MQL5 are stored as
integers of type int, an implicit casting of any enumeration to (int) is supported, and an explicit casting
int back to any enum type is also allowed. In this case, if the value corresponds to one of the elements
of the enumeration, the EnumToString function returns a string with the ID of this element. Otherwise,
the function returns a string of the form ENUM_TYPE::value.
Thus, by looping over integers in the acceptable range and explicitly casting them to an enum type, one
can then analyze the output string EnumToString for the presence of '::' to determine whether the
given integer is an enum member or not.
The StringFind function used here will be presented in the next chapter, just like other string functions.
Let's create the ConversionEnum.mq5 script to test the concept. In it, we implement an auxiliary
function process, which will call the EnumToArray template, report the number of elements in the enum,
and print the resulting array with matches between the enum elements and their values.
template<typename E>
void process(E a)
{
  int result[];
  int n = EnumToArray(a, result, 0, USHORT_MAX);
  Print(typename(E), " Count=", n);
  for(int i = 0; i < n; i++)
  {
    Print(i, " ", EnumToString((E)result[i]), "=", result[i]);
  }
}
As an enumeration for research purposes, we will use the built-in enumeration with the
ENUM_APPLIED_PRICE price types. Inside the function OnStart, let's first make sure that
EnumToString produces strings as described above. So, for the element PRICE_CLOSE, the function will
return the string "PRICE_CLOSE", and for the value (ENUM_APPLIED_PRICE)1 0, which is obviously out
of range, it will return "ENUM_APPLIED_PRICE::1 0".
void OnStart()
{
   PRT(EnumToString(PRICE_CLOSE));            // PRICE_CLOSE
   PRT(EnumToString((ENUM_APPLIED_PRICE)10)); // ENUM_APPLIED_PRICE::10
   
   process((ENUM_APPLIED_PRICE)0);
}
Next, we call the function process for any value cast to ENUM_APPLIED_PRICE (or a variable of that
type) and get the following result:

---

## Page 303

Part 4. Common APIs
303
4.1  Built-in type conversions
ENUM_APPLIED_PRICE Count=7
0 PRICE_CLOSE=1
1 PRICE_OPEN=2
2 PRICE_HIGH=3
3 PRICE_LOW=4
4 PRICE_MEDIAN=5
5 PRICE_TYPICAL=6
6 PRICE_WEIGHTED=7
Here we see that 7 elements are defined in the enumeration, and the numbering does not start from 0,
as usual, but from 1  (PRICE_CLOSE). Knowing the values associated with the elements allows in some
cases to optimize the writing of algorithms.
4.1 .7 Type complex
The built-in type complex is a structure with two fields of type double:
struct complex 
{ 
   double      real;   // real part 
   double      imag;   // imaginary part 
};
This structure is described in the type conversion section because it "converts" two double numbers
into a new entity, in something similar to how structures are turned into byte arrays, and vice versa.
Moreover, it would be rather difficult to introduce this type without describing the structures first.
The complex structure does not have a constructor, so complex numbers must be created using an
initialization list.
complex c = {re, im};
For complex numbers, only simple arithmetic and comparison operations are currently available: =, +, -
, *, /, +=, -=, *=, /=, ==, !=. Support for mathematical functions will be added later.
Attention! Complex variables cannot be declared as inputs (using the keyword input) for an MQL
program.
The suffix 'i' is used to describe complex (imaginary parts) constants, for example:
const complex x = 1 - 2i;
const complex y = 0.5i;
In the following example (script Complex.mq5) a complex number is created and squared.

---

## Page 304

Part 4. Common APIs
304
4.1  Built-in type conversions
input double r = 1;
input double i = 2;
   
complex c = {r, i};
   
complex mirror(const complex z)
{
   complex result = {z.imag, z.real}; // swap real and imaginary parts
   return result;
}
   
complex square(const complex z) 
{ 
   return (z * z);
}   
   
void OnStart()
{
   Print(c);
   Print(square(c));
   Print(square(mirror(c)));
}
With default parameters, the script will output the following:
c=(1,2) / ok
square(c)=(-3,4) / ok
square(mirror(c))=(3,4) / ok
Here, the pairs of numbers in parentheses are the string representation of the complex number.
Type complex can be passed by value as a parameter of MQL functions (unlike ordinary structures,
which are passed only by reference). For functions imported from DLL, the type complex should only be
passed by reference.
4.2 Working with strings and symbols
Although computers take their name from the verb "compute", they are equally successful in
processing not only numbers but also any unstructured information, the most famous example of which
is text. In MQL programs, text is also used everywhere, from the names of the programs themselves to
comments in trade orders. To work with the text in MQL5, there is a built-in string type, which allows
you to operate on character sequences of arbitrary length.
To perform typical actions with strings, the MQL5 API provides a wide range of functions that can be
conditionally divided into groups according to their purpose, such as string initialization, their addition,
searching and replacing fragments within strings, converting strings to character arrays, accessing
individual characters, as well as formatting.
Most of the functions in this chapter return an indication of the execution status: success or error. For
functions with result type bool, true is usually a success, and false is an error. For functions with result
type int a value of 0 or -1  can be considered an error: this is stated in the description of each function.
In all these cases, the developer can find out the essence of the problem. To do this, call the

---

## Page 305

Part 4. Common APIs
305
4.2 Working with strings and symbols
GetLastError function and get the specific error code: a list of all codes with explanations is available in
the documentation. It's important to call GetLastError immediately after receiving the error flag
because calling each following instruction in the algorithm can lead to another error.
4.2.1  Initialization and measurement of strings
As we know from the String type section, it is enough to describe in the code a variable of type string,
and it will be ready to go.
For any variable of string type 1 2 bytes are allocated for the service structure which is the internal
representation of the string. The structure contains the memory address (pointer) where the text is
stored, along with some other meta-information. The text itself also requires sufficient memory, but
this buffer is allocated with some less obvious optimizations.
In particular, we can describe a string along with explicit initialization, including an empty literal:
string s = ""; // pointer to the literal containing '\0'
In that case, the pointer will be set directly to the literal, and no memory is allocated for the buffer
(even if the literal is long). Obviously, static memory has already been allocated for the literal, and it
can be used directly. The memory for the buffer will be allocated only if any instruction in the program
changes the contents of the line. For example (note the addition operation '+' is allowed for strings):
int n = 1;
s += (string)n;    // pointer to memory containing "1"'\0'[plus reserve]
From this point on, the string actually contains the text "1 " and, strictly speaking, requires memory for
two characters: the digit "1 " and the implicit terminal zero '\0' (terminator of the string). However, the
system will allocate a larger buffer, with some space reserved.
When we declare a variable without an initial value, it is still implicitly initialized by the compiler, though
in this case with a special NULL value:
string z; // memory for the pointer is not allocated, pointer = NULL
Such a string requires only 1 2 bytes per structure, and the pointer doesn't point anywhere: that's what
NULL stands for.
In future versions of the MQL5 compiler, this behavior may change, and a small area of memory will
always be initially allocated for an empty string, providing some reserved space.
In addition to these internal features, variables of the string type are no different from variables of
other types. However, due to the fact that strings can be variable in length and, more importantly, they
can change their length during the algorithm, this can adversely affect the efficiency of memory
allocation and performance.
For example, if at some point the program needs to add a new word to a string, it may turn out that
there is not enough memory allocated for the string. Then the MQL program execution environment,
imperceptible to the user, will find a new free memory block of increased size and copy the old value
there along with the added word. After that, the old address is replaced by a new one in the line's
service structure.
If there are many such operations, slowdown due to copying can become noticeable, and in addition,
program memory is subject to fragmentation: old small memory areas released after copying form
voids that are not suitable in size for large strings, and therefore lead to waste of memory. Of course,

---

## Page 306

Part 4. Common APIs
306
4.2 Working with strings and symbols
the terminal is able to control such situations and reorganize the memory, but this also comes at a
cost.
The most effective way to solve this problem is to explicitly indicate in advance the size of the buffer for
the string and initialize it using the built-in MQL5 API functions, which we will consider later in this
section.
The basis for this optimization is just that the size of the allocated memory may exceed the current
(and, potentially, the future) length of the string, which is determined by the first null character in the
text. Thus, we can allocate a buffer for 1 00 characters, but from the start put '\0' at the very
beginning, which will give a zero-length string ("").
Naturally, it is assumed that in such cases the programmer can roughly calculate in advance the
expected length of the string or its growth rate.
Since strings in MQL5 are based on double-byte characters (which ensures Unicode support), the size
of the string and buffer in characters should be multiplied by 2 to get the amount of occupied and
allocated memory in bytes.
A general example of using all functions (StringInit.mq5) will be given at the end of the section.
bool StringInit(string &variable, int capacity = 0, ushort character = 0)
The StringInit function is used to initialize (allocate and fill memory) and deinitialize (free memory)
strings. The variable to be processed is passed in the first parameter.
If the capacity parameter is greater than 0, then a buffer (memory area) of the specified size is
allocated for the string and is filled with the symbol character. If the character is 0, then the length of
the string will be zero, because the first character is terminal.
If the capacity parameter is 0, then previously allocated memory is freed. The state of the variable
becomes identical to how it was if just declared without initialization (the pointer to the buffer is NULL).
More simply, the same can be done by setting the string variable to NULL.
The function returns a success indicator (true) or errors (false).
bool StringReserve(string &variable, uint capacity)
The StringReserve function increases or decreases the buffer size of the string variable, at least up to
the number of characters specified in the capacity parameter. If the capacity value is less than the
current string length, the function does nothing. In fact, the buffer size may be larger than requested:
the environment does this for reasons of efficiency in future manipulations with the string. Thus, if the
function is called with a reduced value for the buffer, it can ignore the request and still return true ("no
errors").
The current buffer size can be obtained using the function StringBufferLen (see below).
On success, the function returns true, otherwise — false.
Unlike StringInit the StringReserve function does not change the contents of the string and does not fill
it with characters.

---

## Page 307

Part 4. Common APIs
307
4.2 Working with strings and symbols
bool StringFill(string &variable, ushort character)
The StringFill function fills the specified variable string with the character character for its entire
current length (up to the first zero). If a buffer is allocated for a string, the modification is done in-
place, without intermediate newline and copy operations.
The function returns a success indicator (true) or errors (false).
int StringBufferLen(const string &variable)
The function returns the size of the buffer allocated for the variable string.
Note that for a literal-initialized string, no buffer is initially allocated because the pointer points to the
literal. Therefore, the function will return 0 even though the length of the StringLen string (see below)
may be more.
The value -1  means that the line belongs to the client terminal and cannot be changed.
bool StringSetLength(string &variable, uint length)
The function sets the specified length in characters length for the variable string. The value of the
length must not be greater than the current length of the string. In other words, the function only
allows you to shorten the string, but not lengthen it. The length of the string is increased automatically
when the StringAdd function is called, or the addition operation '+' is performed.
The equivalent of the function StringSetLength is the call StringSetCharacter(variable, length, 0) (see
section Working with symbols and code pages).
If a buffer has already been allocated for the string before the function call, the function does not
change it. If the string did not have a buffer (it was pointing to a literal), decreasing the length results
in allocating a new buffer and copying the shortened string into it.
The function returns true or false in case of success or failure, respectively.
int StringLen(const string text)
The function returns the number of characters in the string text. Terminal zero is not taken into
account.
Please note that the parameter is passed by value, so you can calculate the length of strings not only
in variables but also for any other intermediate values: calculation results or literals.
The StringInit.mq5 script has been created to demonstrate the above functions. It uses a special
version of the PRT macro, PRTE, which parses the result of an expression into true or false, and in the
case of the latter additionally outputs an error code:
#define PRTE(A) Print(#A, "=", (A) ? "true" : "false:" + (string)GetLastError())
For debug output to the log of a string and its current metrics (line length and buffer size), the StrOut
function is implemented:
void StrOut(const string &s)
{
   Print("'", s, "' [", StringLen(s), "] ", StringBufferLen(s));
}
It uses the built-in StringLen and StringBufferLen functions.

---

## Page 308

Part 4. Common APIs
308
4.2 Working with strings and symbols
The test script performs a series of actions on a string in OnStart:
void OnStart()
{
   string s = "message";
   StrOut(s);
   PRTE(StringReserve(s, 100)); // ok, but we get a buffer larger than requested: 260
   StrOut(s);
   PRTE(StringReserve(s, 500)); // ok, buffer is increased to 500
   StrOut(s);
   PRTE(StringSetLength(s, 4)); // ok: string is shortened
   StrOut(s);
   s += "age";
   PRTE(StringReserve(s, 100)); // ok: buffer remains at 500
   StrOut(s);
   PRTE(StringSetLength(s, 8)); // no: string lengthening is not supported
   StrOut(s);                   //     via StringSetLength
   PRTE(StringInit(s, 8, '$')); // ok: line increased by padding
   StrOut(s);                   //     buffer remains the same
   PRTE(StringFill(s, 0));      // ok: string collapsed to empty because
   StrOut(s);                   //     was filled with 0s, the buffer is the same
   PRTE(StringInit(s, 0));      // ok: line is zeroed, including buffer
                                // we could just write s = NULL;
   StrOut(s);
}
The script will log the following messages:
'message' [7] 0
StringReserve(s,100)=true
'message' [7] 260
StringReserve(s,500)=true
'message' [7] 500
StringSetLength(s,4)=true
'mess' [4] 500
StringReserve(s,10)=true
'message' [7] 500
StringSetLength(s,8)=false:5035
'message' [7] 500
StringInit(s,8,'$')=true
'$$$$$$$$' [8] 500
StringFill(s,0)=true
'' [0] 500
StringInit(s,0)=true
'' [0] 0
Please note that the call StringSetLength with increased string length ended with error 5035
(ERR_STRING_SMALL_LEN).

---

## Page 309

Part 4. Common APIs
309
4.2 Working with strings and symbols
4.2.2 String concatenation
Concatenation of strings is probably the most common string operation. In MQL5, it can be done using
the '+' or '+=' operators. The first operator concatenates two strings (the operands to the left and
right of the '+') and creates a temporary concatenated string that can be assigned to a target variable
or passed to another part of an expression (such as a function call). The second operator appends the
string to the right of the operator '+=' to the string (variable) to the left of this operator.
In addition to this, the MQL5 API provides a couple of functions for composing strings from other
strings or elements of other types.
Examples of using functions are given in the script StringAdd.mq5, which is considered after their
description.
bool StringAdd(string &variable, const string addition)
The function appends the specified addition string to the end of a string variable variable. Whenever
possible, the system uses the available buffer of the string variable (if its size is enough for the
combined result) without re-allocating memory or copying strings.
The function is equivalent to the operator variable += addition. Time costs and memory consumption
are about the same.
The function returns true in case of success and false in case of error.
int StringConcatenate(string &variable, void argument1 , void argument2 [, void argumentI...])
The function converts two or more arguments of built-in types to a string representation and
concatenates them in the variable string. The arguments are passed starting from the second
parameter of the function. Arrays, structures, objects, pointers are not supported as arguments.
The number of arguments must be between 2 and 63.
String arguments are added to the resulting variable as is.
Arguments of type double are converted with maximum precision (up to 1 6 significant digits), and
scientific notation with exponent can be chosen if it turns out to be more compact. Arguments of type
float are displayed with 5 characters.
Values of type datetime are converted to a string with all date and time fields ("YYYY.MM.DD
hh:mm:ss").
Enumerations, single-byte and double-byte characters are output as integers.
Values of type color are displayed as a trio of "R,G,B" components or a color name (if available in the
list of standard web colors).
When converting type bool the strings "true" or "false" are used.
The function StringConcatenate returns the length of the resulting string.
StringConcatenate is designed to build a string from other sources (variables, expressions) other
than the receiving variable. It is not recommended to use StringConcatenate to concatenate new
chunks of data to the same row by calling StringConcatenate(variable, variable, ...). This function call
is not optimized and is extremely slow compared to the operator '+' and StringAdd.

---

## Page 310

Part 4. Common APIs
31 0
4.2 Working with strings and symbols
Functions StringAdd and StringConcatenate are tested in the StringAdd.mq5 script, which uses the
PRTE macro and the helper function StrOut from the previous section.
void OnStart()
{
   string s = "message";
   StrOut(s);
   PRTE(StringAdd(s, "r"));
   StrOut(s);
   PRTE(StringConcatenate(s, M_PI * 100, " ", clrBlue, PRICE_CLOSE));
   StrOut(s);
}
As a result of its execution, the following lines are displayed in the log:
'message' [7] 0
StringAdd(s,r)=true
'messager' [8] 260
StringConcatenate(s,M_PI*100, ,clrBlue,PRICE_CLOSE)=true
'314.1592653589793 clrBlue1' [26] 260
The script also includes the header file StringBenchmark.mqh with the class benchmark. It provides a
framework for derived classes implemented in the script to measure the performance of various string
addition methods. In particular, they make sure that adding strings using the operator '+' and the
function StringAdd are comparable. This material is left for independent study.
Additionally, the book comes with the script StringReserve.mq5: it makes a visual comparison of the
speed of adding strings depending on the use or non-use of the buffer (StringReserve).
4.2.3 String comparison
To compare strings in MQL5, you can use the standard comparison operators, in particular '==', '!=',
'>', '<'. All such operators conduct comparisons in a character-by-character, case-sensitive manner.
Each character has a Unicode code which is an integer of type ushort. Accordingly, first the codes of
the first characters of two strings are compared, then the codes of the second ones, and so on until the
first mismatch or the end of one of the strings is reached.
For example, the string "ABC" is less than "abc", because the codes of uppercase letters in the
character table are lower than the codes of the corresponding lowercase letters (on the first character
we already get that "A" < "a"). If strings have matching characters at the beginning, but one of them
is longer than the other, then the longer string is considered to be greater ("ABCD" > "ABC").
Such string relationships form the lexicographic order. When the string "A" is less than the string
"B" ("A" < "B"), "A" is said to precede "B".
To get familiar with the character codes, you can use the standard Windows application "Character
Table". In it, the characters are arranged in order of increasing codes. In addition to the general
Unicode table, which includes many national languages, there are code pages: ANSI standard tables
with single-byte character codes – they differ for each language or group of languages. We will
explore this issue in more detail in the section Working with symbols and code pages.
The initial part of the character tables with codes from 0 to 1 27 is the same for all languages. This part
is shown in the following table.

---

## Page 311

Part 4. Common APIs
31 1 
4.2 Working with strings and symbols
ASCII character code table
To obtain the character code, take the hexadecimal digit on the left (the line number in which the
character is located) and add the number on top (the column number in which the character is
located): the result is a hexadecimal number. For example, for '!' there is 2 on the left and 1  on the
top, which means the character code is 0x21 , or 33 in decimal.
Codes up to 32 are control codes. Among them, you can find, in particular, tabulation (code 0x9), line
feed (code 0xA), and carriage return (code 0xD).
A pair of characters 0xD 0xA following one another is used in Windows text files to break to a new line.
We got acquainted with the corresponding MQL5 literals in the Character types section: 0xA can be
denoted as '\n' and 0xD as '\r'. The tabulation 0x9 also has its own representation: '\t'.
The MQL5 API provides the StringCompare function, which allows you to disable case sensitivity when
comparing strings.
int StringCompare(const string &string1 , const string &string2, const bool case_sensitive = true)
The function compares two strings and returns one of three values: +1  if the first string is "greater
than" the second; 0 if strings are "equal"; -1  if the first string is "less than" the second one. The
concepts of "greater than", "less than" and "equal to" depend on the case_ sensitive parameter.
When the case_ sensitive parameter equals true (which is the default), the comparison is case-sensitive,
with uppercase letters being considered greater than similar lowercase ones. This is the reverse of the
standard lexicographic order according to character codes.
When case-sensitive, the StringCompare function uses an order of uppercase and lowercase letters
that is different from the lexicographical order. For example, we know that the relation "A" < "a" is
true, in which the operator '<' is guided by character codes. Therefore, capitalized words should
appear in the hypothetical dictionary (array) before words with the same lowercase letter.
However, when comparing "A" and "a" using the StringCompare("A", "a") function, we get +1  which
means "A" is greater than "a". Thus, in the sorted dictionary, words starting with lowercase letters
will come first, and only after them will come words with capital letters.
In other words, the function ranks the strings alphabetically. Besides that, in the case sensitivity mode,
an additional rule applies: if there are strings that differ only in case, those that have uppercase letters
follow their counterparts with lowercase letters (at the same positions in the word).

---

## Page 312

Part 4. Common APIs
31 2
4.2 Working with strings and symbols
If the case_ sensitive parameter equals false, the letters are case insensitive, so the strings "A" and "a"
are considered equal, and the function returns 0.
You can check different comparison results by the StringCompare function and by the operator using
the StringCompare.mq5 script.
void OnStart()
{
   PRT(StringCompare("A", "a"));        // 1, which means "A" > "a" (!)
   PRT(StringCompare("A", "a", false)); // 0, which means "A" == "a"
   PRT("A" > "a");                      // false,   "A" < "a"
   
   PRT(StringCompare("x", "y"));        // -1, which means "x" < "y"
   PRT("x" > "y");                      // false,    "x" < "y"
   ...
}
In the Function Templates section, we have created a templated quicksort algorithm. Let's transform it
into a template class and use it for several sorting options: using comparison operators, as well as using
the StringCompare function both with and without case sensitivity enabled. Let's put the new
QuickSortT class in the QuickSortT.mqh header file and connect it to the test script
StringCompare.mq5.
The sorting API has remained almost unchanged.

---

## Page 313

Part 4. Common APIs
31 3
4.2 Working with strings and symbols
template<typename T>
class QuickSortT
{
public:
   void Swap(T &array[], const int i, const int j)
   {
      ...
   }
   
   virtual int Compare(T &a, T &b)
   {
      return a > b ? +1 : (a < b ? -1 : 0);
   }
   
   void QuickSort(T &array[], const int start = 0, int end = INT_MAX)
   {
      ...
         for(int i = start; i <= end; i++)
         {
            //if(!(array[i] > array[end]))
            if(Compare(array[i], array[end]) <= 0)
            {
               Swap(array, i, pivot++);
            }
         }
      ...
   }
};
The main difference is that we have added a virtual method Compare, which by default contains a
comparison using the '>' and '<' operators, and returns +1 , -1 , or 0 in the same way as
StringCompare. The Compare method is now used in the QuickSort method instead of a simple
comparison and must be overridden in child classes in order to use the StringCompare function or any
other way of comparison.
In particular, in the StringCompare.mq5 file, we implement the following "comparator" class derived
from QuickSortT<string>:

---

## Page 314

Part 4. Common APIs
31 4
4.2 Working with strings and symbols
class SortingStringCompare : public QuickSortT<string>
{
   const bool caseEnabled;
public:
   SortingStringCompare(const bool sensitivity = true) :
      caseEnabled(sensitivity) { }
      
   virtual int Compare(string &a, string &b) override
   {
      return StringCompare(a, b, caseEnabled);
   }
};
The constructor receives 1  parameter, which specifies string comparison sign taking into account
(true) or ignoring (false) the register. The string comparison itself is done in the redefined virtual
method Compare which calls the function StringCompare with the given arguments and setting.
To test sorting, we need a set of strings that combines uppercase and lowercase letters. We can
generate it ourselves: it is enough to develop a class that performs permutations (with repetition) of
characters from a predefined set (alphabet) for a given set length (string). For example, you can limit
yourself to the small alphabet "abcABC", that is, three fist English letters in both cases, and generate
all possible strings of 2 characters from them.
The class PermutationGenerator is supplied in the file PermutationGenerator.mqh and left for
independent study. Here we present only its public interface.
class PermutationGenerator
{
public:
   struct Result
   {
      int indices[]; // indexes of elements in each position of the set, i.e.
   };                // for example, the numbers of the letters of the "alphabet" in each position of the string 
   PermutationGenerator(const int length, const int elements);
   SimpleArray<Result> *run();
};
When creating a generator object, you must specify the length of the generated sets length (in our
case, this will be the length of the strings, i.e., 2) and the number of different elements from which the
sets will be composed (in our case, this is the number of unique letters, that is, 6). With such input
data, 6 * 6 = 36 variants of lines should be obtained.
The process itself is carried out by run method. A template class is used to return an array with results
SimpleArray, which we discussed in the Method Templates section. In this case, it is parameterized by
the structure type result.
The call of the generator and the actual creation of strings in accordance with the array of
permutations received from it (in the form of letter indices at each position for all possible strings) is
performed in the auxiliary function GenerateStringList.

---

## Page 315

Part 4. Common APIs
31 5
4.2 Working with strings and symbols
void GenerateStringList(const string symbols, const int len, string &result[])
{
   const int n = StringLen(symbols); // alphabet length, unique characters
   PermutationGenerator g(len, n);
   SimpleArray<PermutationGenerator::Result> *r = g.run();
   ArrayResize(result, r.size());
   // loop through all received character permutations
   for(int i = 0; i < r.size(); ++i)
   {
      string element;
      // loop through all characters in the string
      for(int j = 0; j < len; ++j)
      {
         // add a letter from the alphabet (by its index) to the string
         element += ShortToString(symbols[r[i].indices[j]]);
      }
      result[i] = element;
   }
}
Here we use several functions that are still unfamiliar to us (ArrayResize, ShortToString), but we'll get
to them soon. For now, we should only know that the ShortToString function returns a string consisting
of that single character based on the ushort type character code. Using the operator '+=', we
concatenate each resulting string from such single-character strings. Recall that the operator [] is
defined for strings, so the expression symbols[k] will return the k-th character of the symbols string. Of
course, k can in turn be an integer expression, and here r[i].indices[j ]  is referring to i-th element of the
r array from which the index of the "alphabet" character is read for the j-th position of the string.
Each received string is stored in an array-parameter result.
Let's apply this information in the OnStart function.

---

## Page 316

Part 4. Common APIs
31 6
4.2 Working with strings and symbols
void OnStart()
{
   ...
   string messages[];
   GenerateStringList("abcABC", 2, messages);
   Print("Original data[", ArraySize(messages), "]:");
   ArrayPrint(messages);
   
   Print("Default case-sensitive sorting:");
   QuickSortT<string> sorting;
   sorting.QuickSort(messages);
   ArrayPrint(messages);
   
   Print("StringCompare case-insensitive sorting:");
   SortingStringCompare caseOff(false);
   caseOff.QuickSort(messages);
   ArrayPrint(messages);
   
   Print("StringCompare case-sensitive sorting:");
   SortingStringCompare caseOn(true);
   caseOn.QuickSort(messages);
   ArrayPrint(messages);
}
The script first gets all string options into the messages array and then sorts it in 3 modes: using the
built-in comparison operators, using the StringCompare function in the case-insensitive mode and using
the same function in the case-sensitive mode.
We will get the following log output:
Original data[36]:
[ 0] "aa" "ab" "ac" "aA" "aB" "aC" "ba" "bb" "bc" "bA" "bB" "bC" "ca" "cb" "cc" "cA" "cB" "cC"
[18] "Aa" "Ab" "Ac" "AA" "AB" "AC" "Ba" "Bb" "Bc" "BA" "BB" "BC" "Ca" "Cb" "Cc" "CA" "CB" "CC"
Default case-sensitive sorting:
[ 0] "AA" "AB" "AC" "Aa" "Ab" "Ac" "BA" "BB" "BC" "Ba" "Bb" "Bc" "CA" "CB" "CC" "Ca" "Cb" "Cc"
[18] "aA" "aB" "aC" "aa" "ab" "ac" "bA" "bB" "bC" "ba" "bb" "bc" "cA" "cB" "cC" "ca" "cb" "cc"
StringCompare case-insensitive sorting:
[ 0] "AA" "Aa" "aA" "aa" "AB" "aB" "Ab" "ab" "aC" "AC" "Ac" "ac" "BA" "Ba" "bA" "ba" "BB" "bB"
[18] "Bb" "bb" "bC" "BC" "Bc" "bc" "CA" "Ca" "cA" "ca" "CB" "cB" "Cb" "cb" "cC" "CC" "Cc" "cc"
StringCompare case-sensitive sorting:
[ 0] "aa" "aA" "Aa" "AA" "ab" "aB" "Ab" "AB" "ac" "aC" "Ac" "AC" "ba" "bA" "Ba" "BA" "bb" "bB"
[18] "Bb" "BB" "bc" "bC" "Bc" "BC" "ca" "cA" "Ca" "CA" "cb" "cB" "Cb" "CB" "cc" "cC" "Cc" "CC"
The output shows the differences in these three modes.
4.2.4 Changing the character case and trimming spaces
Working with texts often implies the use of some standard operations, such as converting all characters
to upper or lower case and removing extra empty characters (for example, spaces) at the beginning or
end of a string. For these purposes, the MQL5 API provides four corresponding functions. All of them
modify the string in place, that is, directly in the available buffer (if it is already allocated).

---

## Page 317

Part 4. Common APIs
31 7
4.2 Working with strings and symbols
The input parameter of all functions is a reference to a string, i.e., only variables (not expressions) can
be passed to them, and not constant variables since the functions involve modifying the argument.
The test script for all functions follows the relevant descriptions.
bool StringToLower(string &variable)
bool StringToUpper(string &variable)
The functions convert all characters of the specified string to the appropriate case: StringToLower to
lowercase letters, and StringToUpper to uppercase. This includes support for national languages
available at the Windows system level.
If successful, it returns true. In case of an error, it returns false.
int StringTrimLeft(string &variable)
int StringTrimRight(string &variable)
The function removes carriage return ('\r'), line feed ('\n'), spaces (' '), tabs ('\t') and some other
non-displayable characters at the beginning (for StringTrimLeft) or end (for StringTrimRight) of a string.
If there are empty spaces inside the string (between the displayed characters), they will be preserved.
The function returns the number of characters removed.
The StringModify.mq5 file demonstrates the operation of the above functions.
void OnStart()
{
   string text = "  \tAbCdE F1  ";
               // ↑        ↑  ↑
               // |        |  └2 spaces
               // |        └space
               // └2 spaces and tab
   PRT(StringToLower(text));   // 'true'
   PRT(text);                  // '  \tabcde f1  '
   PRT(StringToUpper(text));   // 'true'
   PRT(text);                  // '  \tABCDE F1  '
   PRT(StringTrimLeft(text));  // '3'
   PRT(text);                  // 'ABCDE F1  '
   PRT(StringTrimRight(text)); // '2'
   PRT(text);                  // 'ABCDE F1'
   PRT(StringTrimRight(text)); // '0'  (there is nothing else to delete)
   PRT(text);                  // 'ABCDE F1'
                               //       ↑
                               //       └the space inside remains
   
   string russian = "Russian text";
   PRT(StringToUpper(russian));  // 'true'
   PRT(russian);                 // 'RUSSIAN TEXT'  
   string german = "straßenführung";
   PRT(StringToUpper(german));   // 'true'
   PRT(german);                  // 'STRAßENFÜHRUNG'
}

---

## Page 318

Part 4. Common APIs
31 8
4.2 Working with strings and symbols
4.2.5 Finding, replacing, and extracting string fragments
Perhaps the most popular operations when working with strings are finding and replacing fragments, as
well as extracting them. In this section, we will study the MQL5 API functions that will help solve these
problems. Examples of their use are summarized in the StringFindReplace.mq5 file.
int StringFind(string value, string wanted, int start = 0)
The function searches for the substring wanted in the string value, starting from the position start. If
the substring is found, the function will return the position where it starts, with the characters in the
string numbered starting from 0. Otherwise, the function will return -1 . Both parameters are passed by
value, which allows processing not only variables but also intermediate results of calculations
(expressions, function calls).
The search is performed based on a strict match of characters, i.e., it is case-sensitive. If you want to
search in a case-insensitive way, you must first convert the source string to a single case using
StringToLower or StringToUpper.
Let's try to count the number of occurrences of the desired substring in the text using StringFind. To
do this, let's write a helper function CountSubstring which will call StringFind in a loop, gradually shifting
the search starting position in the last parameter start. The loop continues as long as new occurrences
of the substring are found.
int CountSubstring(const string value, const string wanted)
{
   // indent back because of the increment at the beginning of the loop
   int cursor = -1;
   int count = -1;
   do
   {
      ++count;
      ++cursor; // search continues from the next position
      // get the position of the next substring, or -1 if there are no matches
      cursor = StringFind(value, wanted, cursor);
   }
   while(cursor > -1);
   return count;
}
It is important to note that the presented implementation looks for substrings that can overlap. This is
because the current position is changed by 1  (++cursor) before it starts looking for the next
occurrence. As a result, when searching for, let's say, the substring "AAA" in the string "AAAAA", 3
matches will be found. The technical requirements for searching may differ from this behavior. In
particular, there is a practice to continue searching after the position where the previously found
fragment ended. In this case, it will be necessary to modify the algorithm so that the cursor moves with
a step equal to StringLen(wanted).
Let's call CountSubstring for different arguments in the OnStart function.

---

## Page 319

Part 4. Common APIs
31 9
4.2 Working with strings and symbols
void OnStart()
{
   string abracadabra = "ABRACADABRA";
   PRT(CountSubstring(abracadabra, "A"));    // 5
   PRT(CountSubstring(abracadabra, "D"));    // 1
   PRT(CountSubstring(abracadabra, "E"));    // 0
   PRT(CountSubstring(abracadabra, "ABRA")); // 2
   ...
}
int StringReplace(string &variable, const string wanted, const string replacement)
The function replaces all found wanted substrings with the replacement substring in the variable string.
The function returns the number of replacements made or -1  in case of an error. The error code can be
obtained by calling the function GetLastError. In particular, these can be out-of-memory errors or the
use of an uninitialized string (NULL) as an argument. The variables and wanted parameters must be
strings of non-zero length.
When an empty string "" is given as the replacement argument, all occurrences of wanted are simply
cut from the original string.
If there were no substitutions, the result of the function is 0.
Let's use the example of StringFindReplace.mq5 to check StringReplace in action.
   string abracadabra = "ABRACADABRA";
   ...
   PRT(StringReplace(abracadabra, "ABRA", "-ABRA-")); // 2
   PRT(StringReplace(abracadabra, "CAD", "-"));      // 1
   PRT(StringReplace(abracadabra, "", "XYZ"));      // -1, error
   PRT(GetLastError());      // 5040, ERR_WRONG_STRING_PARAMETER
   PRT(abracadabra);                              // '-ABRA---ABRA-'
   ...
Next, using the StringReplace function, let's try to execute one of the tasks encountered in the
processing of arbitrary texts. We will try to ensure that a certain separator character is always used as
a single character, i.e., sequences of several such characters must be replaced by one. Typically, this
refers to spaces between words, but there may be other separators in technical data. Let's test our
program for the separator '-'.
We implement the algorithm as a separate function NormalizeSeparatorsByReplace:

---

## Page 320

Part 4. Common APIs
320
4.2 Working with strings and symbols
int NormalizeSeparatorsByReplace(string &value, const ushort separator = ' ')
{
   const string single = ShortToString(separator);
   const string twin = single + single;
   int count = 0;
   int replaced = 0;
   do
   {
      replaced = StringReplace(value, twin, single);
      if(replaced > 0) count += replaced;
   }
   while(replaced > 0);
   return count;
}
The program tries to replace a sequence of two separators with one in a do-while loop, and the loop
continues as long as the StringReplace function returns values greater than 0 (i.e., there is still
something to replace). The function returns the total number of replacements made.
In the function OnStart let's "clear" our inscription from multiple characters '-'.
   ...
   string copy1 = "-" + abracadabra + "-";
   string copy2 = copy1;
   PRT(copy1);                                    // '--ABRA---ABRA--'
   PRT(NormalizeSeparatorsByReplace(copy1, '-')); // 4
   PRT(copy1);                                    // '-ABRA-ABRA-'
   PRT(StringReplace(copy1, "-", ""));            // 1
   PRT(copy1);                                    // 'ABRAABRA'
   ...
int StringSplit(const string value, const ushort separator, string &result[])
The function splits the passed value string into substrings based on the given separator and puts them
into the result array. The function returns the number of received substrings or -1  in case of an error.
If there is no separator in the string, the array will have one element equal to the entire string.
If the source string is empty or NULL, the function will return 0.
To demonstrate the operation of this function, let's solve the previous problem in a new way using
StringSplit. To do this, let's write the function NormalizeSeparatorsBySplit.

---

## Page 321

Part 4. Common APIs
321 
4.2 Working with strings and symbols
int NormalizeSeparatorsBySplit(string &value, const ushort separator = ' ')
{
   const string single = ShortToString(separator);
   
   string elements[];
   const int n = StringSplit(value, separator, elements);
   ArrayPrint(elements); // debug
   
   StringFill(value, 0); // result will replace original string
   
   for(int i = 0; i < n; ++i)
   {
      // empty strings mean delimiters, and we only need to add them
      // if the previous line is not empty (i.e. not a separator either)
      if(elements[i] == "" && (i == 0 || elements[i - 1] != ""))
      {
         value += single;
      }
      else // all other lines are joined together "as is"
      {
         value += elements[i];
      }
   }
   
   return n;
}
When separators occur one after another in the source text, the corresponding element in the output
array StringSplit turns out to be an empty string "". Also, an empty string will be at the beginning of
the array if the text starts with a separator, and at the end of the array if the text ends with the
separator.
To get "cleared" text, you need to add all non-empty strings from the array, "gluing" them with single
separator characters. Moreover, only those empty elements in which the previous element of the array
is also not empty should be converted into a separator.
Of course, this is only one of the possible options for implementing this functionality. Let's check it in
the OnStart function.
   ...
   string copy2 = "-" + abracadabra + "-";        // '--ABRA---ABRA--'
   PRT(NormalizeSeparatorsBySplit(copy2, '-'));   // 8
   // debug output of split array (inside function):
   // ""     ""     "ABRA" ""     ""     "ABRA" ""     ""
   PRT(copy2);                                    // '-ABRA-ABRA-'
string StringSubstr(string value, int start, int length = -1 )
The function extracts from the passed text value a substring starting at the specified position start, of
the length length. The starting position can be from 0 to the length of the string minus 1 . If the length
length is -1  or more than the number of characters from start to the end of the string, the rest of the
string will be extracted in full.

---

## Page 322

Part 4. Common APIs
322
4.2 Working with strings and symbols
The function returns a substring or an empty string if the parameters are incorrect.
Let's see how it works.
   PRT(StringSubstr("ABRACADABRA", 4, 3));        // 'CAD'
   PRT(StringSubstr("ABRACADABRA", 4, 100));      // 'CADABRA'
   PRT(StringSubstr("ABRACADABRA", 4));           // 'CADABRA'
   PRT(StringSubstr("ABRACADABRA", 100));         // ''
4.2.6 Working with symbols and code pages
Since strings are made up of characters, it is sometimes necessary or simply more convenient to
manipulate individual characters or groups of characters in a string at the level of their integer codes.
For example, you need to read or replace characters one at a time or convert them into arrays of
integer codes for transmission over communication protocols or into third-party programming
interfaces of dynamic libraries DLL. In all such cases, passing strings as text can be accompanied by
various difficulties:
• ensuring the correct encoding (of which there are a great many, and the choice of a specific one
depends on the operating system locale, program settings, the configuration of the servers with
which communication is carried out, and much more)
• conversion of national language characters from the local text encoding to Unicode and vice versa
• allocation and deallocation of memory in a unified way
The use of arrays with integer codes (while such use actually produces a binary rather than a textual
representation of the string) simplifies these problems.
The MQL5 API provides a set of functions to operate on individual characters or their groups, taking
into account encoding features.
Strings in MQL5 contain characters in two-byte Unicode encoding. This provides universal support for
the entire variety of national alphabets in a single (but very large) character table. Two bytes allow the
encoding of 65535 elements.
The default character type is ushort. However, if necessary, the string can be converted to a sequence
of single-byte uchar characters in a specific language encoding. This conversion may be accompanied
by the loss of some information (in particular, letters that are not in the localized character table may
"lose" umlauts or even "turn" into some kind of substitute character: depending on the context, it can
be displayed differently, but usually as ' ?' or a square character).
To avoid problems with texts that may contain arbitrary characters, it is recommended that you
always use Unicode. An exception can be made if some external services or programs that should
be integrated with your MQL program do not support Unicode, or if the text is intended from the
beginning to store a limited set of characters (for example, only numbers and Latin letters).
When converting to/from single-byte characters, the MQL5 API uses the ANSI encoding by default,
depending on the current Windows settings. However, the developer can specify a different code table
(see further functions CharArrayToString, StringToCharArray).
Examples of using the functions described below are given in the StringSymbols.mq5 file.

---

## Page 323

Part 4. Common APIs
323
4.2 Working with strings and symbols
bool StringSetCharacter(string &variable, int position, ushort character)
The function changes the character at position to the character value in the passed variable string. The
number must be between 0 and the string length (StringLen) minus 1 .
If the character to be written is 0, it specifies a new line ending (acts as a terminal zero), i.e. the
length of the line becomes equal to position. The size of the buffer allocated for the line does not
change.
If the position parameter is equal to the length of the string and the character being written is not
equal to 0, then the character is added to the string and its length is increased by 1 . This is equivalent
to the expression: variable += ShortToString(character).
The function returns true upon successful completion, or false in case of error.
void OnStart()
{
   string numbers = "0123456789";
   PRT(numbers);
   PRT(StringSetCharacter(numbers, 7, 0));   // cut off at the 7th character
   PRT(numbers);                             // 0123456
   PRT(StringSetCharacter(numbers, StringLen(numbers), '*')); // add '*'
   PRT(numbers);                             // 0123456*
   ...
}
ushort StringGetCharacter(string value, int position)
The function returns the code of the character located at the specified position in the string. The
position number must be between 0 and the string length (StringLen) minus 1 . In case of an error, the
function will return 0.
The function is equivalent to writing using the operator '[]': value[position].
   string numbers = "0123456789";
   PRT(StringGetCharacter(numbers, 5));      // 53 = code '5'
   PRT(numbers[5]);                          // 53 - is the same 
string CharToString(uchar code)
The function converts the ANSI code of a character to a single-character string. Depending on the set
Windows code page, the upper half of the codes (greater than 1 27) can generate different letters (the
character style is different, while the code remains the same). For example, the symbol with the code
0xB8 (1 84 in decimal) denotes a cedilla (lower hook) in Western European languages, while in the
Russian language the letter 'ё' is located here. Here's another example:
   PRT(CharToString(0xA9));   // "©"
  PRT(CharToString(0xE6));   // "æ", "ж", or another character
                             // depending on your Windows locale

---

## Page 324

Part 4. Common APIs
324
4.2 Working with strings and symbols
string ShortToString(ushort code)
The function converts the Unicode code of a character to a single-character string. For the code
parameter, you can use a literal or an integer. For example, the Greek capital letter "sigma" (the sign
of the sum in mathematical formulas) can be specified as 0x3A3 or 'Σ'.
   PRT(ShortToString(0x3A3)); // "Σ"
   PRT(ShortToString('Σ'));   // "Σ"
int StringToShortArray(const string text, ushort &array[], int start = 0, int count = -1 )
The function converts a string to a sequence of ushort character codes that are copied to the specified
location in the array: starting from the element numbered start (0 by default, that is, the beginning of
the array) and in the amount of count.
Please note: the start parameter refers to the position in the array, not in the string. If you want to
convert part of a string, you must first extract it using the StringSubstr function.
If the count parameter is equal to -1  (or WHOLE_ARRAY), all characters up to the end of the string
(including the terminal null) or characters in accordance with the size of the array, if it is a fixed size,
are copied.
In the case of a dynamic array, it will be automatically increased in size if necessary. If the size of a
dynamic array is greater than the length of the string, then the size of the array is not reduced.
To copy characters without a terminating null, you must explicitly call StringLen as the count
argument. Otherwise, the length of the array will be by 1  more than the length of the string (and 0 in
the last element).
The function returns the number of copied characters.

---

## Page 325

Part 4. Common APIs
325
4.2 Working with strings and symbols
   ...
   ushort array1[], array2[]; // dynamic arrays 
   ushort text[5];            // fixed size array 
   string alphabet = "ABCDEАБВГД";
   // copy with the terminal '0'
   PRT(StringToShortArray(alphabet, array1)); // 11
   ArrayPrint(array1); // 65   66   67   68   69 1040 1041 1042 1043 1044    0
   // copy without the terminal '0'
   PRT(StringToShortArray(alphabet, array2, 0, StringLen(alphabet))); // 10
   ArrayPrint(array2); // 65   66   67   68   69 1040 1041 1042 1043 1044
   // copy to a fixed array 
   PRT(StringToShortArray(alphabet, text)); // 5
   ArrayPrint(text); // 65 66 67 68 69
   // copy beyond the previous limits of the array 
   // (elements [11-19] will be random)
   PRT(StringToShortArray(alphabet, array2, 20)); // 11
   ArrayPrint(array2);
   /*
   [ 0]    65    66    67    68    69  1040  1041  1042
         1043  1044     0     0     0     0     0 14245
   [16] 15102 37754 48617 54228    65    66    67    68
           69  1040  1041  1042  1043  1044     0
   */
Note that if the position for copying is beyond the size of the array, then the intermediate elements will
be allocated but not initialized. As a result, they may contain random data (highlighted in yellow above).
string ShortArrayToString(const ushort &array[], int start = 0, int count = -1 )
The function converts part of the array with character codes to a string. The range of array elements is
set by parameters start and count, the starting position, and quantity, respectively. The parameter
start must be between 0 and the number of elements in the array minus 1 . If count is equal to -1  (or
WHOLE_ARRAY) all elements up to the end of the array or up to the first null are copied.
Using the same example from StringSymbols.mq5, let's try to convert an array into the array2 string,
which has a size of 30.
   ...
   string s = ShortArrayToString(array2, 0, 30);
   PRT(s); // "ABCDEАБВГД", additional random characters may appear here
Because in the array array2 the string "ABCDEABCD" was copied twice, and specifically, firstly to the
very beginning, and the second time –at offset 20, the intermediate characters will be random and able
to form a longer string than we did.
int StringToCharArray(const string text, uchar &array[], int start = 0, int count = -1 , uint codepage =
CP_ACP)
The function converts the text string into a sequence of single-byte characters that are copied to the
specified location in the array: starting from the element numbered start (0 by default, that is, the
beginning of the array) and in the amount of count. The copying process converts characters from
Unicode to the selected code page codepage – by default, CP_ACP, which means the language of the
Windows operating system (more on this below).

---

## Page 326

Part 4. Common APIs
326
4.2 Working with strings and symbols
If the count parameter is equal to -1  (or WHOLE_ARRAY), all characters up to the end of the string
(including the terminal null) or in accordance with the size of the array, if it is a fixed size, are copied.
In the case of a dynamic array, it will be automatically increased in size if necessary. If the size of a
dynamic array is greater than the length of the string, then the size of the array is not reduced.
To copy characters without a terminating null, you must explicitly call StringLen as an argument count.
The function returns the number of copied characters.
See the list of valid code pages for the parameter codepage in the documentation. Here are some of
the widely used ANSI code pages:
Language
Code
Central European Latin
1 250
Cyrillic
1 251 
Western European Latin
1 252
Greek
1 253
Turkish
1 254
Hebrew
1 255
Arab
1 256
Baltic
1 257
Thus, on computers with Western European languages, CP_ACP is 1 252, and, for example, on
computers with Russian, it is 1 251 .
During the conversion process, some characters may be converted with loss of information, since the
Unicode table is much larger than ANSI (each ANSI code table has 256 characters).
In this regard, CP_UTF8 is of particular importance among all the CP_*** constants. It allows national
characters to be properly preserved by variable-length encoding: the resulting array still stores bytes,
but each national character can span multiple bytes, written in a special format. Because of this, the
length of the array can be significantly larger than the length of the string. UTF-8 encoding is widely
used on the Internet and in various software. Incidentally, UTF stands for Unicode Transformation
Format, and there are other modifications, notably UTF-1 6 and UTF-32.
We will consider an example for StringToCharArray after we get acquainted with the "inverse" function
CharArrayToString: their work must be demonstrated in conjunction.
string CharArrayToString(const uchar &array[], int start = 0, int count = -1 , uint codepage =
CP_ACP)
The function converts an array of bytes or part of it into a string. The array must contain characters in
a specific encoding. The range of array elements is set by parameters start and count, the starting
position, and quantity, respectively. The parameter start must be between 0 and the number of
elements in the array. When count is equal to -1  (or WHOLE_ARRAY) all elements up to the end of the
array or up to the first null are copied.

---

## Page 327

Part 4. Common APIs
327
4.2 Working with strings and symbols
Let's see how the functions StringToCharArray and CharArrayToString work with different national
characters with different code page settings. A test script StringCodepages.mq5 has been prepared for
this.
Two lines will be used as the test subjects - in Russian and German:
void OnStart()
{
   Print("Locales");
   uchar bytes1[], bytes2[];
   string german = "straßenführung";
   string russian = "Russian text";
   ...
We will copy them into arrays bytes1  and bytes2 and then restore them to strings.
First, let's convert the German text using the European code page 1 252.
   ...
   StringToCharArray(german, bytes1, 0, WHOLE_ARRAY, 1252);
   ArrayPrint(bytes1);
   // 115 116 114  97 223 101 110 102 252 104 114 117 110 103   0
On European copies of Windows, this is equivalent to a simpler function call with default parameters,
because there CP_ACP = 1 252:
   StringToCharArray(german, bytes1);
Then we restore the text from the array with the following call and make sure that everything matches
the original:
   ...
   PRT(CharArrayToString(bytes1, 0, WHOLE_ARRAY, 1252));
   // CharArrayToString(bytes1,0,WHOLE_ARRAY,1252)='straßenführung'
Now let's try to convert the Russian text in the same European encoding (or you can call
StringToCharArray(english, bytes2) in the Windows environment where CP_ACP is set to 1 252 as the
default code page):
   ...
   StringToCharArray(russian, bytes2, 0, WHOLE_ARRAY, 1252);
   ArrayPrint(bytes2);
   // 63 63 63 63 63 63 63 32 63 63 63 63 63  0
Here you can already see that there was a problem during the conversion because 1 252 does not have
Cyrillic. Restoring a string from an array clearly shows the essence:
   ...
   PRT(CharArrayToString(bytes2, 0, WHOLE_ARRAY, 1252));
   // CharArrayToString(bytes2,0,WHOLE_ARRAY,1252)='??????? ?????'
Let's repeat the experiment in a conditional Russian environment, i.e., we will convert both strings back
and forth using the Cyrillic code page 1 251 .

---

## Page 328

Part 4. Common APIs
328
4.2 Working with strings and symbols
   ...
   StringToCharArray(russian, bytes2, 0, WHOLE_ARRAY, 1251);
   // on Russian Windows, this call is equivalent to a simpler one
   // StringToCharArray(russian, bytes2);
   // because CP_ACP = 1251
   ArrayPrint(bytes2); // this time the character codes are meaningful
   // 208 243 241 241 234 232 233  32 210 229 234 241 242   0
   
   // restore the string and make sure it matches the original
   PRT(CharArrayToString(bytes2, 0, WHOLE_ARRAY, 1251));
   // CharArrayToString(bytes2,0,WHOLE_ARRAY,1251)='Русский Текст'
   
   // and for the German text...
   StringToCharArray(german, bytes1, 0, WHOLE_ARRAY, 1251);
   ArrayPrint(bytes1);
   // 115 116 114  97  63 101 110 102 117 104 114 117 110 103   0
   // if we compare this content of bytes1 with the previous version,
   // it's easy to see that a couple of characters are affected; here's what happened:
   // 115 116 114  97 223 101 110 102 252 104 114 117 110 103   0
   
   // restore the string to see the differences visually:
   PRT(CharArrayToString(bytes1, 0, WHOLE_ARRAY, 1251));
   // CharArrayToString(bytes1,0,WHOLE_ARRAY,1251)='stra?enfuhrung'
   // specific German characters were corrupted
Thus, the fragility of single-byte encodings is evident.
Finally, let's enable the CP_UTF8 encoding for both test strings. This part of the example will work
stably regardless of Windows settings.
   ...
   StringToCharArray(german, bytes1, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayPrint(bytes1);
   // 115 116 114  97 195 159 101 110 102 195 188 104 114 117 110 103   0
   PRT(CharArrayToString(bytes1, 0, WHOLE_ARRAY, CP_UTF8));
   // CharArrayToString(bytes1,0,WHOLE_ARRAY,CP_UTF8)='straßenführung'
   
   StringToCharArray(russian, bytes2, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayPrint(bytes2);
   // 208 160 209 131 209 129 209 129 208 186 208 184 208 185
   //  32 208 162 208 181 208 186 209 129 209 130   0
   PRT(CharArrayToString(bytes2, 0, WHOLE_ARRAY, CP_UTF8));
   // CharArrayToString(bytes2,0,WHOLE_ARRAY,CP_UTF8)='Русский Текст'
Note that both of the UTF-8 encoded strings required larger arrays than ANSI ones. Moreover, the
array with the Russian text has actually become 2 times longer, because all letters now occupy 2
bytes. Those who wish can find details in open sources on how exactly the UTF-8 encoding works. In
the context of this book, it is important for us that the MQL5 API provides ready-made functions to
work with.

---

## Page 329

Part 4. Common APIs
329
4.2 Working with strings and symbols
4.2.7 Universal formatted data output to a string
When generating a string to display to the user, to save to a file, or to send over the Internet, it may be
necessary to include the values of several variables of different types in it. This problem can be solved
by explicitly casting all variables to the type (string) and adding the resulting strings, but in this case,
the MQL code instruction will be long and difficult to understand. It would probably be more convenient
to use the StringConcatenate function, but this method does not completely solve the problem.
The fact is that a string usually contains not only variables, but also some text inserts that act as
connecting links and provide the correct structure of the overall message. It turns out that pieces of
formatting text are mixed with variables. This kind of code is hard to maintain, which goes against one
of the well-known principles of programming: the separation of content and presentation.
There is a special solution for this problem: the StringFormat function.
The same scheme applies to another MQL5 API function: PrintFormat.
string StringFormat(const string format, ...)
The function converts arbitrary built-in type arguments to a string according to the specified format.
The first parameter is the template of the string to be prepared, in which the places for inserting
variables are indicated in a special way and the format of their output is determined. These control
commands may be interspersed with plain text, which is copied to the output string unchanged. The
following function parameters, separated by commas, list all the variables in the order and types that
are reserved for them in the template.
Interaction of the format string and StringFormat arguments
Each variable insertion point in a string is marked with a format specifier: the character '%', after
which several settings can be specified.
The format string is parsed from left to right. When the first specifier (if any) is encountered, the value
of the first parameter after the format string is converted and added to the resulting string according
to the specified settings. The second specifier causes the second parameter to be converted and
printed, and so on, until the end of the format string. All other characters in the pattern between the
specifiers are copied unchanged into the resulting string.
The template may not contain any specifier, that is, it can be a simple string. In this case, you need to
pass a dummy argument to the function in addition to the string (the argument will not be placed in the
string).
If you want to display the percent sign in the template, then you should write it twice in a row %%. If
the % sign is not doubled, then the next few characters following % are always parsed as a specifier.

---

## Page 330

Part 4. Common APIs
330
4.2 Working with strings and symbols
A mandatory attribute of a specifier is a symbol that indicates the expected type and interpretation of
the next function argument. Let's conditionally call this symbol T. Then, in the simplest case, one
format specifier looks like %T.
In a generalized form, the specifier can consist of several more fields (optional fields are indicated in
square brackets):
%[Z][W][.P][M]T
Each field performs its function and takes one of the allowed values. Next, we will gradually consider all
the fields.
Type T
For integers, the following characters can be used as T, with an explanation of how the corresponding
numbers are displayed in the string:
• c – Unicode character
• C – ANSI character
• d, i – signed decimal
• o – unsigned octal
• u – unsigned decimal
• x – unsigned hexadecimal (lowercase)
• X – unsigned hexadecimal (capital letters)
Recall that according to the method of internal data storage, integer types also include built-in MQL5
types datetime, color, bool and enumerations.
For real numbers, the following symbols are applicable as T:
• e – scientific format with exponent (lowercase 'e')
• E – scientific format with exponent (capital 'E')
• f – normal format
• g – analog of f or e (the most compact form is chosen)
• G – analog of f or E (the most compact form is chosen)
• a – scientific format with exponent, hexadecimal (lowercase)
• A – scientific format with exponent, hexadecimal (capital letters)
Finally, there is only one version of the T character available for strings: s.
Size of integers M
For integer types, you can additionally explicitly specify the size of the variable in bytes by prefixing T
with one of the following characters or combinations of them (we have generalized them under the
letter M):
• h – 2 bytes (short, ushort)
• l (lowercase L) – 4 bytes (int, uint)
• I32 (capital i) – 4 bytes (int, uint)
• ll (two lowercase Ls) – 8 bytes (long)
• I64 (capital i) – 8 bytes (long, ulong)

---

## Page 331

Part 4. Common APIs
331 
4.2 Working with strings and symbols
Width W
The W field is a non-negative decimal number that specifies the minimum number of character spaces
available for the formatted value. If the value of the variable fits into fewer characters, then the
corresponding number of spaces is added to the left or right. The left or right side is selected depending
on the alignment (see the flag further '–' in the Z field). If the '0' flag is present, the corresponding
number of zeros is added in front of the output value. If the number of characters to be output is
greater than the specified width, then the width setting is ignored and the output value is not
truncated.
If an asterisk '*' is specified as the width, then the width of the output value should be specified in the
list of passed parameters. It should be a value of type int at the position preceding the variable being
formatted.
Precision P
The P field also contains a non-negative decimal number and is always preceded by a dot '.'. For
integer T, this field specifies the minimum number of significant digits. If the value fits in fewer digits, it
is prepended with zeros.
For real numbers, P specifies the number of decimal places (default is 6), except for the g and G
specifiers, for which P is the total number of significant digits (mantissa and decimal).
For a string, P specifies the number of characters to display. If the string length exceeds the precision
value, then the string will be shown as truncated.
If the asterisk '*' is specified as the precision, it is treated in the same way as for the width but
controls the precision.
Fixed width and/or precision, together with the right-alignment, makes it possible to display values in a
neat column.
Flags Z
Finally, the Z field describes the flags:
• - (minus) – left alignment within the specified width (in the absence of the flag, right alignment is
done);
• + (plus) – unconditional display of a '+' or '-' sign before the value (without this flag, only '-' is
displayed for negative values);
• 0 – zeros are added before the output value if it is less than the specified width;
• (space) – a space is placed before the displayed value if it is signed and positive;
• # – controls the display of octal and hexadecimal number prefixes in formats o, x or X (for example,
for the format x prefix "0x" is added before the displayed number, for the format X – prefix "0X"),
decimal point in real numbers (formats e, E, a or A) with a zero fractional part, and some other
nuances.
You can learn more about the possibilities of formatted output to a string in the documentation.
The total number of function parameters cannot exceed 64.
If the number of arguments passed to the function is greater than the number of specifiers, then the
extra arguments are omitted.

---

## Page 332

Part 4. Common APIs
332
4.2 Working with strings and symbols
If the number of specifiers in the format string is greater than the arguments, then the system will try
to display zeros instead of missing data, but a text warning ("missing string parameter") will be
embedded for string specifiers.
If the type of the value does not match the type of the corresponding specifier, the system will try to
read the data from the variable in accordance with the format and display the resulting value (it may
look strange due to a misinterpretation of the internal bit representation of the real data). In the case
of strings, a warning ("non-string passed") may be embedded in the result.
Let's test the function with the script StringFormat.mq5.
First, let's try different options for T and data type specifier.
PRT(StringFormat("[Infinity Sign] Unicode (ok): %c; ANSI (overflow): %C", 
   '∞', '∞'));
PRT(StringFormat("short (ok): %hi, short (overflow): %hi", 
   SHORT_MAX, INT_MAX));
PRT(StringFormat("int (ok): %i, int (overflow): %i", 
   INT_MAX, LONG_MAX));
PRT(StringFormat("long (ok): %lli, long (overflow): %i", 
   LONG_MAX, LONG_MAX));
PRT(StringFormat("ulong (ok): %llu, long signed (overflow): %lli", 
   ULONG_MAX, ULONG_MAX));
Both correct and incorrect specifiers are represented here (incorrect ones come second in each
instruction and are marked with the word "overflow" since the value passed does not fit in the format
type).
Here's what happens in the log (the breaks of long lines here and below are made for publication):
StringFormat(Plain string,0)='Plain string'
StringFormat([Infinity Sign] Unicode: %c; ANSI: %C,'∞','∞')=
   '[Infinity Sign] Unicode (ok): ∞; ANSI (overflow):  '
StringFormat(short (ok): %hi, short (overflow): %hi,SHORT_MAX,INT_MAX)=
   'short (ok): 32767, short (overflow): -1'
StringFormat(int (ok): %i, int (overflow): %i,INT_MAX,LONG_MAX)=
   'int (ok): 2147483647, int (overflow): -1'
StringFormat(long (ok): %lli, long (overflow): %i,LONG_MAX,LONG_MAX)=
   'long (ok): 9223372036854775807, long (overflow): -1'
StringFormat(ulong (ok): %llu, long signed (overflow): %lli,ULONG_MAX,ULONG_MAX)=
   'ulong (ok): 18446744073709551615, long signed (overflow): -1'
All of the following instructions are correct:
PRT(StringFormat("ulong (ok): %I64u", ULONG_MAX));
PRT(StringFormat("ulong (HEX): %I64X, ulong (hex): %I64x", 
   1234567890123456, 1234567890123456));
PRT(StringFormat("double PI: %f", M_PI));
PRT(StringFormat("double PI: %e", M_PI));
PRT(StringFormat("double PI: %g", M_PI));
PRT(StringFormat("double PI: %a", M_PI));
PRT(StringFormat("string: %s", "ABCDEFGHIJ"));
The result of their work is shown below:

---

## Page 333

Part 4. Common APIs
333
4.2 Working with strings and symbols
StringFormat(ulong (ok): %I64u,ULONG_MAX)=
   'ulong (ok): 18446744073709551615'
StringFormat(ulong (HEX): %I64X, ulong (hex): %I64x,1234567890123456,1234567890123456)=
   'ulong (HEX): 462D53C8ABAC0, ulong (hex): 462d53c8abac0'
StringFormat(double PI: %f,M_PI)='double PI: 3.141593'
StringFormat(double PI: %e,M_PI)='double PI: 3.141593e+00'
StringFormat(double PI: %g,M_PI)='double PI: 3.14159'
StringFormat(double PI: %a,M_PI)='double PI: 0x1.921fb54442d18p+1'
StringFormat(string: %s,ABCDEFGHIJ)='string: ABCDEFGHIJ'
Now let's look at the various modifiers.
With right alignment (by default) and a fixed field width (number of characters), we can use different
options for padding the resulting string on the left: with a space or zeros. In addition, for any alignment,
you can enable or disable the explicit indication of the sign of the value (so that not only minus is
displayed for negative, but also plus for positive).
PRT(StringFormat("space padding: %10i", SHORT_MAX));
PRT(StringFormat("0-padding: %010i", SHORT_MAX));
PRT(StringFormat("with sign: %+10i", SHORT_MAX));
PRT(StringFormat("precision: %.10i", SHORT_MAX));
We get the following in the log:
StringFormat(space padding: %10i,SHORT_MAX)='space padding:      32767'
StringFormat(0-padding: %010i,SHORT_MAX)='0-padding: 0000032767'
StringFormat(with sign: %+10i,SHORT_MAX)='with sign:     +32767'
StringFormat(precision: %.10i,SHORT_MAX)='precision: 0000032767'
To align to the left, you must use the '-' (minus) flag, the addition of the string to the specified width
occurs on the right:
PRT(StringFormat("no sign (default): %-10i", SHORT_MAX));
PRT(StringFormat("with sign: %+-10i", SHORT_MAX));
Result:
StringFormat(no sign (default): %-10i,SHORT_MAX)='no sign (default): 32767     '
StringFormat(with sign: %+-10i,SHORT_MAX)='with sign: +32767    '
If necessary, we can show or hide the sign of the value (by default, only minus is displayed for negative
values), add a space for positive values, and thus ensure the same formatting when you need to display
variables in a column:
PRT(StringFormat("default: %i", SHORT_MAX));  // standard
PRT(StringFormat("default: %i", SHORT_MIN));
PRT(StringFormat("space  : % i", SHORT_MAX)); // extra space for positive
PRT(StringFormat("space  : % i", SHORT_MIN));
PRT(StringFormat("sign   : %+i", SHORT_MAX)); // force sign output
PRT(StringFormat("sign   : %+i", SHORT_MIN));
Here's what it looks like in the log:

---

## Page 334

Part 4. Common APIs
334
4.2 Working with strings and symbols
StringFormat(default: %i,SHORT_MAX)='default: 32767'
StringFormat(default: %i,SHORT_MIN)='default: -32768'
StringFormat(space  : % i,SHORT_MAX)='space  :  32767'
StringFormat(space  : % i,SHORT_MIN)='space  : -32768'
StringFormat(sign   : %+i,SHORT_MAX)='sign   : +32767'
StringFormat(sign   : %+i,SHORT_MIN)='sign   : -32768'
Now let's compare how width and precision affect real numbers.
PRT(StringFormat("double PI: %15.10f", M_PI));
PRT(StringFormat("double PI: %15.10e", M_PI));
PRT(StringFormat("double PI: %15.10g", M_PI));
PRT(StringFormat("double PI: %15.10a", M_PI));
   
// default precision = 6
PRT(StringFormat("double PI: %15f", M_PI));
PRT(StringFormat("double PI: %15e", M_PI));
PRT(StringFormat("double PI: %15g", M_PI));
PRT(StringFormat("double PI: %15a", M_PI));
Result:
StringFormat(double PI: %15.10f,M_PI)='double PI:    3.1415926536'
StringFormat(double PI: %15.10e,M_PI)='double PI: 3.1415926536e+00'
StringFormat(double PI: %15.10g,M_PI)='double PI:     3.141592654'
StringFormat(double PI: %15.10a,M_PI)='double PI: 0x1.921fb54443p+1'
StringFormat(double PI: %15f,M_PI)='double PI:        3.141593'
StringFormat(double PI: %15e,M_PI)='double PI:    3.141593e+00'
StringFormat(double PI: %15g,M_PI)='double PI:         3.14159'
StringFormat(double PI: %15a,M_PI)='double PI: 0x1.921fb54442d18p+1'
In the explicit width is not specified, the values are output without padding with spaces.
PRT(StringFormat("double PI: %.10f", M_PI));
PRT(StringFormat("double PI: %.10e", M_PI));
PRT(StringFormat("double PI: %.10g", M_PI));
PRT(StringFormat("double PI: %.10a", M_PI));
Result:
StringFormat(double PI: %.10f,M_PI)='double PI: 3.1415926536'
StringFormat(double PI: %.10e,M_PI)='double PI: 3.1415926536e+00'
StringFormat(double PI: %.10g,M_PI)='double PI: 3.141592654'
StringFormat(double PI: %.10a,M_PI)='double PI: 0x1.921fb54443p+1'
Setting the width and precision of values using the sign '*' and based on additional function arguments
is performed as follows:
PRT(StringFormat("double PI: %*.*f", 12, 5, M_PI));
PRT(StringFormat("string: %*s", 15, "ABCDEFGHIJ"));
PRT(StringFormat("string: %-*s", 15, "ABCDEFGHIJ"));
Please note that 1  or 2 integer type values are passed before the output value, according to the
number of asterisks '*' in the specifier: you can control the precision and the width separately or both
together.

---

## Page 335

Part 4. Common APIs
335
4.2 Working with strings and symbols
StringFormat(double PI: %*.*f,12,5,M_PI)='double PI:      3.14159'
StringFormat(string: %*s,15,ABCDEFGHIJ)='string:      ABCDEFGHIJ'
StringFormat(string: %-*s,15,ABCDEFGHIJ)='string: ABCDEFGHIJ     '
Finally, let's look at a few common formatting errors.
PRT(StringFormat("string: %s %d %f %s", "ABCDEFGHIJ"));
PRT(StringFormat("string vs int: %d", "ABCDEFGHIJ"));
PRT(StringFormat("double vs int: %d", M_PI));
PRT(StringFormat("string vs double: %s", M_PI));
The first instruction has more specifiers than arguments. In other cases, the types of specifiers and
passed values do not match. As a result, we get the following output:
StringFormat(string: %s %d %f %s,ABCDEFGHIJ)=
   'string: ABCDEFGHIJ 0 0.000000 (missed string parameter)'
StringFormat(string vs int: %d,ABCDEFGHIJ)='string vs int: 0'
StringFormat(double vs int: %d,M_PI)='double vs int: 1413754136'
StringFormat(string vs double: %s,M_PI)=
   'string vs double: (non-string passed)'
Having a single format string in every StringFormat function call allows you to use it, in particular, to
translate the external interface of programs and messages into different languages: simply download
and substitute into StringFormat various format strings (prepared in advance) depending on user
preferences or terminal settings.
4.3 Working with arrays
It is difficult to imagine any program and especially one related to trading, without arrays. We have
already studied the general principles of describing and using arrays in the Arrays chapter. They are
organically complemented by a set of built-in functions for working with arrays.
Some of them provide ready-made implementations of the most commonly used array operations, such
as finding the maximum and minimum, sorting, inserting, and deleting elements.
However, there are a number of functions without which it is impossible to use arrays of specific types.
In particular, a dynamic array must first allocate memory before working with it, and arrays with data
for indicator buffers (we will study this MQL program type in Part 5 of the book) use a special order of
element indexing, set by a special function.
And we will begin looking at functions for working with arrays with the output operation to the log. We
already saw it in previous chapters of the book and will be useful in many subsequent ones.
Since MQL5 arrays can be multidimensional (from 1  to 4 dimensions), we will need to refer to the
dimension numbers further in the text. We will call them numbers, starting with the first, which is
more familiar geometrically and which emphasizes the fact that an array must have at least one
dimension (even if it is empty). However, array elements for each dimension are numbered, as is
customary in MQL5 (and in many other programming languages), from zero. Thus, for an array
described as array[5][1 0], the first dimension is 5 and the second is 1 0.

---

## Page 336

Part 4. Common APIs
336
4.3 Working with arrays
4.3.1  Logging arrays
Printing variables, arrays, and messages about the status of an MQL program to the log is the simplest
means for informing the user, debugging, and diagnosing problems. As for the array, we can implement
element-wise printing using the Print function which we already know from demo scripts. We will
formally describe it a little later, in the section on interaction with the user.
However, it is more convenient to entrust the whole routine related to iteration over elements and their
accurate formatting to the MQL5 environment. The API provides a special ArrayPrint function for this
purpose.
We have already seen examples of working with this function in the Using arrays section. Now let's talk
about its capabilities in more detail.
void ArrayPrint(const void &array[], uint digits = _Digits, const string separator = NULL,
   ulong start = 0, ulong count = WHOLE_ARRAY,
   ulong flags = AR R AYPR IN T_H E AD E R  |  AR R AYPR IN T_IN D E X |  AR R AYPR IN T_L IM IT |  AR R AYPR IN T_D ATE  | 
AR R AYPR IN T_SE C ON D S)
The function logs an array using the specified settings. The array must be one of the built-in types or a
simple structure type. A simple structure is a structure with fields of built-in types, with the exception
of strings and dynamic arrays. The presence of class objects and pointers in the composition of the
structure takes it out of the simple category.
The array must have a dimension of 1  or 2. The formatting automatically adjusts to the array
configuration and, if possible, displays it in a visual form (see below). Despite the fact that MQL5
supports arrays with dimensions of up to 4, the function does not display arrays with 3 or more
dimensions, because it is difficult to represent them in a "flat" form. This happens without generating
errors at the program compilation or execution step.
All parameters except the first one can be omitted, and default values are defined for them.
The digits parameter is used for arrays of real numbers and for numeric fields of structures. It sets the
number of displayed characters in the fractional part of numbers. The default value is one of the
predefined chart variables, namely _ Digits which is the number of decimal places in the current chart's
symbol price.
The separating character separator is used to designate columns when displaying fields in an array of
structures. With the default value (NULL), the function uses a space as a separator.
The start and count parameters set the number of the starting element and the number of elements to
be printed, respectively. By default, the function prints the entire array, but the result can be
additionally affected by the presence of the ARRAYPRINT_LIMIT flag (see below).
The flags parameter accepts a combination of flags that control various display features. Here are
some of them:
• ARRAYPRINT_HEADER outputs the header with the names of the fields of the structure before the
array of structures; it does not affect arrays of non-structures.
• ARRAYPRINT_INDEX outputs indexes of elements by dimensions (for one-dimensional arrays,
indexes are displayed on the left, for two-dimensional arrays they are displayed on the left and
above).
• ARRAYPRINT_LIMIT is used for large arrays, and the output is limited to the first hundred and last
hundred records (this limit is enabled by default).

---

## Page 337

Part 4. Common APIs
337
4.3 Working with arrays
• ARRAYPRINT_DATE is used for values of the datetime type to display the date.
• ARRAYPRINT_MINUTES is used for values of the datetime type to display the time to the nearest
minute.
• ARRAYPRINT_SECONDS is used for values of the datetime type to display the time to the nearest
second.
Values of the datetime type are output by default in the format ARRAYPRINT_DATE | 
ARRAYPRINT_SECONDS.
Values of type color are output in hexadecimal format.
Enumeration values are displayed as integers.
The function does not output nested arrays, structures, and pointers to objects. Three dots are
displayed instead of those.
The ArrayPrint.mq5 script demonstrates how the function works. 
The OnStart function provides definitions of several arrays (one-, two- and three-dimensional), which
are output using ArrayPrint (with default settings).
void OnStart()
{
   int array1D[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
   double array2D[][5] = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}};
   double array3D[][3][5] =
   {
      {{ 1,  2,  3,  4,  5}, { 6,  7,  8,  9, 10}, {11, 12, 13, 14, 15}},
      {{16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {26, 27, 28, 29, 30}},
   };
   
   Print("array1D");
   ArrayPrint(array1D);
   Print("array2D");
   ArrayPrint(array2D);
   Print("array3D");
   ArrayPrint(array3D);
   ...
}
We will get the following lines in the log:
array1D
 1  2  3  4  5  6  7  8  9 10
array2D
         [,0]     [,1]     [,2]     [,3]     [,4]
[0,]  1.00000  2.00000  3.00000  4.00000  5.00000
[1,]  6.00000  7.00000  8.00000  9.00000 10.00000
array3D
The array1 D array is not large enough (it fits in one row), so indexes are not shown for it.
The array2D array has multiple rows (indexes), and therefore their indexes are displayed
(ARRAYPRINT_INDEX is enabled by default).

---

## Page 338

Part 4. Common APIs
338
4.3 Working with arrays
Please note that since the script was run on the EURUSD chart with five-digit prices, _ Digits=5, which
affects the formatting of values of type double.
The array3D array is ignored: no rows were output for it.
Additionally, the Pair and SimpleStruct structures are defined in the script:
struct Pair
{
   int x, y;
};
   
struct SimpleStruct
{
   double value;
   datetime time;
   int count;
   ENUM_APPLIED_PRICE price;
   color clr;
   string details;
   void *ptr;
   Pair pair;
};
SimpleStruct contains fields of built-in types, a pointer to void, as well as a field of type Pair. 
In the OnStart function, an array of type SimpleStruct is created and output using ArrayPrint in two
modes: with default settings and with custom ones (the number of digits after the "comma" is 3, the
separator is ";", the format for datetime is date only).
void OnStart()
{
   ...
   SimpleStruct simple[] =
   {
      { 12.57839, D'2021.07.23 11:15', 22345, PRICE_MEDIAN, clrBlue, "text message"},
      {135.82949, D'2021.06.20 23:45', 8569, PRICE_TYPICAL, clrAzure},
      { 1087.576, D'2021.05.15 10:01:30', -3298, PRICE_WEIGHTED, clrYellow, "note"},
   };
   Print("SimpleStruct (default)");
   ArrayPrint(simple);
   
   Print("SimpleStruct (custom)");
   ArrayPrint(simple, 3, ";", 0, WHOLE_ARRAY, ARRAYPRINT_DATE);
}
This produces the following result:

---

## Page 339

Part 4. Common APIs
339
4.3 Working with arrays
SimpleStruct (default)
       [value]              [time] [count] [type]    [clr]      [details] [ptr] [pair]
[0]   12.57839 2021.07.23 11:15:00   22345      5 00FF0000 "text message"   ...    ...
[1]  135.82949 2021.06.20 23:45:00    8569      6 00FFFFF0 null             ...    ...
[2] 1087.57600 2021.05.15 10:01:30   -3298      7 0000FFFF "note"           ...    ...
SimpleStruct (custom)
  12.578;2021.07.23;  22345;     5;00FF0000;"text message";  ...;   ...
 135.829;2021.06.20;   8569;     6;00FFFFF0;null          ;  ...;   ...
1087.576;2021.05.15;  -3298;     7;0000FFFF;"note"        ;  ...;   ...
Please note that the log that we use in this case and in the previous sections is generated in the
terminal and is available to the user in the tab Experts of the Toolbox window. However, in the future
we will get acquainted with the tester, which provides the same execution environment for certain
types of MQL programs (indicators and Expert Advisors) as the terminal itself. If they are launched
in the tester, the ArrayPrint function and other related functions, which are described in the section
User interaction, will output messages to the log of testing agents.
Until now, we have worked, and will continue to work for some time, only with scripts, and they can
only be executed in the terminal.
4.3.2 Dynamic arrays
Dynamic arrays can change their size during program execution at the request of the programmer.
Let's remember that to describe a dynamic array, you should leave the first pair of brackets after the
array identifier empty. MQL5 requires that all subsequent dimensions (if there are more than one) must
have a fixed size specified with a constant.
It is impossible to dynamically increase the number of elements for any dimension "older" than the first
one. In addition, due to the strict size description, arrays have a "square" shape, i.e., for example, it is
impossible to construct a two-dimensional array with columns or rows of different lengths. If any of
these restrictions are critical for the implementation of the algorithm, you should use not standard
MQL5 arrays, but your own structures or classes written in MQL5.
Note that if an array does not have a size in the first dimension, but does have an initialization list that
allows you to determine the size, then such an array is a fixed-size array, not a dynamic one.
For example, in the previous section, we used the array1 D array:
int array1D[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
Because of the initialization list, its size is known to the compiler, and therefore the array is fixed.
Unlike this simple example, it is not always easy to determine whether a particular array in a real
program is dynamic. In particular, an array can be passed as a parameter into a function. However, it
may be important to know if an array is dynamic because memory can be manually allocated by calling
ArrayResize only for such arrays.
In such cases, the ArrayIsDynamic function allows you to determine the type of the array.
Let's consider some technical descriptions of functions for working with dynamic arrays and then test
them using the ArrayDynamic.mq5 script.

---

## Page 340

Part 4. Common APIs
340
4.3 Working with arrays
bool ArrayIsDynamic(const void &array[])
The function checks if the passed array is dynamic. An array can be of any allowed dimension from 1  to
4. Array elements can be of any type.
The function returns true for a dynamic array, or false in other cases (fixed array, or array with
timeseries, controlled by the terminal itself or by the indicator).
int ArrayResize(void &array[], int size, int reserve = 0)
The function sets the new size in the first dimension of the dynamic array. An array can be of any
allowed dimension from 1  to 4. Array elements can be of any type.
If the reserve parameter is greater than zero, memory is allocated for the array with a reserve for the
specified number of elements. This makes can increase the speed of the program which has many
consecutive function calls. Until the new requested size of the array exceeds the current one taking
into account the reserve, there will be no physical memory reallocation and new elements will be taken
from the reserve.
The function returns the new size of the array if its modification was successful, or -1  in case of an
error.
If the function is applied to a fixed array or timeseries, its size does not change. In these cases, if the
requested size is less than or equal to the current size of the array, the function will return the value of
the size parameter, otherwise, it will return -1 .
When increasing the size of an already existing array, all the data of its elements is preserved. The
added elements are not initialized with anything and may contain arbitrary incorrect data ("garbage").
Setting the array size to 0, ArrayResize(array, 0), does not release the memory actually allocated for it,
including a possible reserve. Such a call will only reset the metadata for the array. This is done for the
purpose of optimizing future operations with the array. To force memory release, use ArrayFree (see
below).
It is important to understand that the reserve parameter is not used every time the function is called,
but only at those moments when the reallocation of memory is actually performed, i.e., when the
requested size exceeds the current capacity of the array including the reserve. To visually show how
this works, we will create an incomplete copy of the internal array object and implement the twin
function ArrayResize for it, and also the analogs ArrayFree and ArraySize, to have a complete toolkit.

---

## Page 341

Part 4. Common APIs
341 
4.3 Working with arrays
template<typename T>
struct DynArray
{
   int size;
   int capacity;
   T memory[];
};
template<typename T>
int DynArraySize(DynArray<T> &array)
{
   return array.size;
}
template<typename T>
void DynArrayFree(DynArray<T> &array)
{
   ArrayFree(array.memory);
   ZeroMemory(array);
}
template<typename T>
int DynArrayResize(DynArray<T> &array, int size, int reserve = 0)
{
   if(size > array.capacity)
   {
      static int temp;
      temp = array.capacity;
      long ul = (long)GetMicrosecondCount();
      array.capacity = ArrayResize(array.memory, size + reserve);
      array.size = MathMin(size, array.capacity);
      ul -= (long)GetMicrosecondCount();
      PrintFormat("Reallocation: [%d] -> [%d], done in %d µs", 
         temp, array.capacity, -ul);
   }
   else
   {
      array.size = size;
   }
   return array.size;
}
An advantage of the DynArrayResize function compared to the built-in ArrayResize is in that that here
we insert a debug printing for those situations when the internal capacity of the array is reallocated.
Now we can take the standard example for the ArrayResize function from the MQL5 documentation and
replace the built-in function calls with "self-made" analogs with the "Dyn" prefix. The modified result is
presented in the script ArrayCapacity.mq5.

---

## Page 342

Part 4. Common APIs
342
4.3 Working with arrays
void OnStart()
{
   ulong start = GetTickCount();
   ulong now;
   int   count = 0;
   
   DynArray<double> a;
   
 // fast option with memory reservation
   Print("--- Test Fast: ArrayResize(arr,100000,100000)");
   
   DynArrayResize(a, 100000, 100000);
   
   for(int i = 1; i <= 300000 && !IsStopped(); i++)
   {
 // set the new size and reserve to 100000 elements
      DynArrayResize(a, i, 100000);
 // on "round" iterations, show the size of the array and the elapsed time
      if(DynArraySize(a) % 100000 == 0)
      {
         now = GetTickCount();
         count++;
         PrintFormat("%d. ArraySize(arr)=%d Time=%d ms", 
            count, DynArraySize(a), (now - start));
         start = now;
      }
   }
   DynArrayFree(a);
   
 // now this is a slow option without redundancy (with less redundancy)
   count = 0;
   start = GetTickCount();
   Print("---- Test Slow: ArrayResize(slow,100000)");
   
   DynArrayResize(a, 100000, 100000);
   
   for(int i = 1; i <= 300000 && !IsStopped(); i++)
   {
 // set new size but with 100 times smaller margin: 1000
      DynArrayResize(a, i, 1000);
 // on "round" iterations, show the size of the array and the elapsed time
      if(DynArraySize(a) % 100000 == 0)
      {
         now = GetTickCount();
         count++;
         PrintFormat("%d. ArraySize(arr)=%d Time=%d ms", 
            count, DynArraySize(a), (now - start));
         start = now;
      }
   }
}

---

## Page 343

Part 4. Common APIs
343
4.3 Working with arrays
The only significant difference is the following: in the slow version, the call ArrayResize(a, i) is replaced
by a more moderate one DynArrayResize(a, i, 1 000), that is, the redistribution is requested not at every
iteration, but at every 1 000th (otherwise the log will be overfilled with messages).
After running the script, we will see the following timing in the log (absolute time intervals depend on
your computer, but we are interested in the difference between performance variants with and without
the reserve):
--- Test Fast: ArrayResize(arr,100000,100000)
Reallocation: [0] -> [200000], done in 17 µs
1. ArraySize(arr)=100000 Time=0 ms
2. ArraySize(arr)=200000 Time=0 ms
Reallocation: [200000] -> [300001], done in 2296 µs
3. ArraySize(arr)=300000 Time=0 ms
---- Test Slow: ArrayResize(slow,100000)
Reallocation: [0] -> [200000], done in 21 µs
1. ArraySize(arr)=100000 Time=0 ms
2. ArraySize(arr)=200000 Time=0 ms
Reallocation: [200000] -> [201001], done in 1838 µs
Reallocation: [201001] -> [202002], done in 1994 µs
Reallocation: [202002] -> [203003], done in 1677 µs
Reallocation: [203003] -> [204004], done in 1983 µs
Reallocation: [204004] -> [205005], done in 1637 µs
...
Reallocation: [295095] -> [296096], done in 2921 µs
Reallocation: [296096] -> [297097], done in 2189 µs
Reallocation: [297097] -> [298098], done in 2152 µs
Reallocation: [298098] -> [299099], done in 2767 µs
Reallocation: [299099] -> [300100], done in 2115 µs
3. ArraySize(arr)=300000 Time=219 ms
The time gain is significant. In addition, we see at which iterations and how the real capacity of the
array (reserve) is changed.
void ArrayFree(void &array[])
The function releases all the memory of the passed dynamic array (including the possible reserve set
using the third parameter of the function ArrayResize) and sets the size of its first dimension to zero.
In theory, arrays in MQL5 release memory automatically when the execution of the algorithm in the
current block ends. It doesn't matter if an array is defined locally (within functions) or globally, whether
it is fixed or dynamic, as the system will free the memory itself in any case, without requiring explicit
actions from the programmer.
Thus, it is not necessary to call this function. However, there are situations when an array is used in an
algorithm to re-fill with something from scratch, i.e., it needs to be freed before each filling. Then this
feature might come in handy.
Keep in mind that if the array elements contain pointers to dynamically allocated objects, the function
does not delete them: the programmer must call delete for them (see below).
Let's test the functions discussed above: ArrayIsDynamic, ArrayResize, ArrayFree.

---

## Page 344

Part 4. Common APIs
344
4.3 Working with arrays
In the ArrayDynamic.mq5 script, the ArrayExtend function is written, which increases the size of the
dynamic array by 1  and writes the passed value to the new element.
template<typename T>
void ArrayExtend(T &array[], const T value)
{
   if(ArrayIsDynamic(array))
   {
      const int n = ArraySize(array);
      ArrayResize(array, n + 1);
      array[n] = (T)value;
   }
}
The ArrayIsDynamic function is used to make sure that the array is only updated if it is dynamic. This is
done in a conditional statement. The ArrayResize function allows you to change the size of the array,
and the ArraySize function is used to find out the current size (it will be discussed in the next section).
In the main function of the script, we will apply ArrayExtend for arrays of different categories: dynamic
and fixed.
void OnStart()
{
   int dynamic[];
   int fixed[10] = {}; // padding with zeros
   
   PRT(ArrayResize(fixed, 0)); // warning: not applicable for fixed array
   
   for(int i = 0; i < 10; ++i)
   {
      ArrayExtend(dynamic, (i + 1) * (i + 1));
      ArrayExtend(fixed, (i + 1) * (i + 1));
   }
   
   Print("Filled");
   ArrayPrint(dynamic);
   ArrayPrint(fixed);
   
   ArrayFree(dynamic);
   ArrayFree(fixed); // warning: not applicable for fixed array
   
   Print("Free Up");
   ArrayPrint(dynamic); // outputs nothing
   ArrayPrint(fixed);
   ...
}
In the code lines calling the functions that cannot be used for fixed arrays, the compiler generates a
"cannot be used for static allocated array" warning. It is important to note that there are no such
warnings inside the ArrayExtend function because an array of any category can be passed to the
function. That is why we check this using ArrayIsDynamic.

---

## Page 345

Part 4. Common APIs
345
4.3 Working with arrays
After a loop in OnStart, the dynamic array will expand to 1 0 and get the elements equal to the squared
indices. The fixed array will remain filled with zeros and will not change size.
Freeing a fixed array with ArrayFree will have no effect, and the dynamic array will actually be deleted.
In this case, the last attempt to print it will not produce any lines in the log.
Let's look at the script execution result.
   ArrayResize(fixed,0)=0
   Filled   
     1   4   9  16  25  36  49  64  81 100
   0 0 0 0 0 0 0 0 0 0
   Free Up
   0 0 0 0 0 0 0 0 0 0
Of particular interest are dynamic arrays with pointers to objects. Let's define a simple dummy class
Dummy and create an array of pointers to such objects.
class Dummy
{
};
void OnStart()
{
   ...
   Dummy *dummies[] = {};
   ArrayExtend(dummies, new Dummy());
   ArrayFree(dummies);
}
After extending the dummy array with a new pointer, we free it with ArrayFree, but there are entries in
the terminal log indicating that the object was left in memory.
1 undeleted objects left
1 object of type Dummy left
24 bytes of leaked memory
The fact is that the function manages only the memory that is allocated for the array. In this case, this
memory held one pointer, but what it points to does not belong to the array. In other words, if the
array contains pointers to "external" objects, then you need to take care of them yourself. For
example:
for(int i = 0; i < ArraySize(dummies); ++i)
{
   delete dummies[i];
}
This deletion must be started before calling ArrayFree.
To shorten the entry, you can use the following macros (loop over elements, call delete for each of
them):

---

## Page 346

Part 4. Common APIs
346
4.3 Working with arrays
#define FORALL(A) for(int _iterator_ = 0; _iterator_ < ArraySize(A); ++_iterator_)
#define FREE(P) { if(CheckPointer(P) == POINTER_DYNAMIC) delete (P); }
#define CALLALL(A, CALL) FORALL(A) { CALL(A[_iterator_]) }
Then deletion of pointers is simplified to the following notation:
   ...
   CALLALL(dummies, FREE);
   ArrayFree(dummies);
As an alternative solution, you can use a pointer wrapper class like AutoPtr, which we discussed in the
section Object type templates. Then the array should be declared with the type AutoPtr. Since the
array will store wrapper objects, not pointers, when the array is cleared, the destructors for each
"wrapper" will be automatically called, and the pointer memory will in turn be freed from them.
4.3.3 Array measurement
One of the main characteristics of an array is its size, that is, the total number of elements in it. It is
important to note that for multidimensional arrays, the size is the product of the lengths of all its
dimensions.
For fixed arrays, you can calculate their size at compile stage using the sizeof operator-based language
construct:
sizeof(array) / sizeof(type)
where array is an identifier, and type is the array type.
For example, if an array is defined in the code fixed:
int fixed[][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
then its size is:
int n = sizeof(fixed) / sizeof(int); // 8
For dynamic arrays, this rule does not work, since the sizeof operator always generates the same size
of the internal dynamic array object: 52 bytes.
Note that in functions, all array parameters are represented internally as dynamic array wrapper
objects. This is done so that an array with any method of memory allocation, including a fixed one,
can be passed to the function. That's why sizeof(array) will return 52 for the parameter array, even
if a fixed size array was passed through it.
The presence of "wrappers" affects only sizeof. The ArrayIsDynamic function always correctly
determines the category of the actual argument passed through the parameter array.
To get the size of any array at the stage of program execution, use the ArraySize function.
int ArraySize(const void &array[])
The function returns the total number of elements in the array. The dimension and type of the array
can be any. For a one-dimensional array, the function call is similar to ArrayRange(array, 0) (see
below).
If the array was distributed with a reserve (the third parameter of the ArrayResize function), its value is
not taken into account.

---

## Page 347

Part 4. Common APIs
347
4.3 Working with arrays
Until memory is allocated for the dynamic array using ArrayResize, the ArraySize function will return 0.
Also, the size becomes zero after calling ArrayFree for the array.
int ArrayRange(const void &array[], int dimension)
The ArrayRange function returns the number of elements in the specified array dimension. The
dimension and type of the array can be any. Parameter dimension must be between 0 and the number
of array dimensions minus 1 . Index 0 corresponds to the first dimension, index 1  to the second, and so
on.
Product of all values of ArrayRange(array, i) with i running over all dimensions gives ArraySize(array).
Let's see the examples of the functions described above (see file ArraySize.mq5).
void OnStart()
{
   int dynamic[];
   int fixed[][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
   
   PRT(sizeof(fixed) / sizeof(int));   // 8
   PRT(ArraySize(fixed));              // 8
   
   ArrayResize(dynamic, 10);
   
   PRT(sizeof(dynamic) / sizeof(int)); // 13 (incorrect)
   PRT(ArraySize(dynamic));            // 10
   
   PRT(ArrayRange(fixed, 0));          // 2
   PRT(ArrayRange(fixed, 1));          // 4
   
   PRT(ArrayRange(dynamic, 0));        // 10
   PRT(ArrayRange(dynamic, 1));        // 0
   int size = 1;
   for(int i = 0; i < 2; ++i)
   {
      size *= ArrayRange(fixed, i);
   }
   PRT(size == ArraySize(fixed));      // true
}
4.3.4 Initializing and populating arrays
Describing an array with an initialization list is possible only for arrays of a fixed size. Dynamic arrays
can be populated only after allocating memory for them by the function ArrayResize. They are
populated using the ArrayInitialize or ArrayFill functions. They are also useful in a program when you
want to bulk-replace values in fixed arrays or time series.
Examples of using the functions are given after their description.
int ArrayInitialize(type &array[], type value)
The function sets all array elements to the specified value. Only arrays of built-in numeric types are
supported (char, uchar, short, ushort, int, uint, long, ulong, bool, color, datetime, float, double). String,

---

## Page 348

Part 4. Common APIs
348
4.3 Working with arrays
structure and pointer arrays cannot be filled in this way: they will need to implement their own
initialization functions. An array can be multidimensional.
The function returns the number of elements.
If the dynamic array is allocated with a reserve (the third parameter of the ArrayResize function), then
the reserve is not initialized.
If, after the array is initialized, its size is increased using ArrayResize, the added elements will not be
automatically set to value. They can be populated using the ArrayFill function.
void ArrayFill(type &array[], int start, int count, type value)
The function fills a numeric array or part of it with a specified value. Part of the array is given by
parameters start and count, which denote the initial number of the element and the number of
elements to be filled, respectively.
It does not matter to the function whether the numbering order of the array elements is set like in
timeseries or not: this property is ignored. In other words, the elements of an array are always counted
from its beginning to its end.
For a multidimensional array, the start parameter can be obtained by converting the coordinates in all
dimensions into a through index for an equivalent one-dimensional array. So, for a two-dimensional
array, the elements with the 0th index in the first dimension are located in memory first, then there will
be the elements with the index 1  in the first dimension, and so on. The formula to calculate start is as
follows:
start = D1 * N2 + D2
where D1  and D2 are the indexes for the first and second dimensions, respectively, N2 is the number of
elements for the second dimension. D2 changes from 0 to (N2-1 ), D1  changes from 0 to (N1 -1 ). For
example, in an array array[3][4] the element with indexes [1 ][3] is the seventh one in a row, and
therefore the call ArrayFill(array, 7, 2, ...) will fill two elements:array[1 ][3] and following after him
array[2][0]. On the diagram, this can be depicted as follows (each cell contains a through index of the
element):
      [][0]  [][1]  [][2]  [][3]
[0][]    0      1      2      3
[1][]    4      5      6      7
[2][]    8      9     10     11
The ArrayFill.mq5 script provides examples of using the aforementioned functions.

---

## Page 349

Part 4. Common APIs
349
4.3 Working with arrays
void OnStart()
{
   int dynamic[];
   int fixed[][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
   
   PRT(ArrayInitialize(fixed, -1));
   ArrayPrint(fixed);
   ArrayFill(fixed, 3, 4, +1);
   ArrayPrint(fixed);
   
   PRT(ArrayResize(dynamic, 10, 50));
   PRT(ArrayInitialize(dynamic, 0));
   ArrayPrint(dynamic);
   PRT(ArrayResize(dynamic, 50));
   ArrayPrint(dynamic);
   ArrayFill(dynamic, 10, 40, 0);
   ArrayPrint(dynamic);
}
Here's what a possible result looks like (random data in uninitialized elements of a dynamic array will be
different):
ArrayInitialize(fixed,-1)=8
    [,0][,1][,2][,3]
[0,]  -1  -1  -1  -1
[1,]  -1  -1  -1  -1
    [,0][,1][,2][,3]
[0,]  -1  -1  -1   1
[1,]   1   1   1  -1
ArrayResize(dynamic,10,50)=10
ArrayInitialize(dynamic,0)=10
0 0 0 0 0 0 0 0 0 0
ArrayResize(dynamic,50)=50
[ 0]           0           0           0           0           0
               0           0           0           0           0
[10] -1402885947  -727144693   699739629   172950740 -1326090126
           47384           0           0     4194184           0
[20]           2           0           2           0           0
               0           0  1765933056  2084602885 -1956758056
[30]    73910037 -1937061701          56           0          56
               0     1048601  1979187200       10851           0
[40]           0           0           0  -685178880 -1720475236
       782716519 -1462194191  1434596297   415166825 -1944066819
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
4.3.5 Copying and editing arrays
In this section, we'll learn how to use built-in functions to insert and remove array elements, change
their order, and copy entire arrays.

---

## Page 350

Part 4. Common APIs
350
4.3 Working with arrays
bool ArrayInsert(void &target[], const void &source[], uint to, uint from = 0, uint count =
WHOLE_ARRAY)
The function inserts the specified number of elements from the source array 'source' into the
destination target array. A position for insertion into the target array is set by the index in the to
parameter. The starting index of the element at which to start copying from the source array is given
by the index from. The WHOLE_ARRAY constant ((uint)-1 ) in the parameter count specifies the transfer
of all elements of the source array.
All indexes and counts are relative to the first dimension of the arrays. In other words, for
multidimensional arrays, the insertion is performed not by individual elements, but by the entire
configuration described by the "higher" dimensions. For example, for a two-dimensional array, the value
1  in the parameter count means inserting a vector of length equal to the second dimension (see the
example).
Due to this, the target array and the source array must have the same configurations. Otherwise, an
error will occur and copying will fail. For one-dimensional arrays, this is not a limitation, but for
multidimensional arrays, it is necessary to observe the equality of sizes in dimensions above the first
one. In particular, elements from the array [][4] cannot be inserted into the array [][5] and vice versa.
The function is applicable only for arrays of fixed or dynamic size. Editing timeseries (arrays with time
series) cannot be performed with this function. It is prohibited to specify in the parameters target and
source the same array.
When inserted into a fixed array, new elements shift existing elements to the right and displace count of
the rightmost elements to the outside of the array. The to parameter must have a value between 0 and
the size of the array minus 1 .
When inserted into a dynamic array, the old elements are also shifted to the right, but they do not
disappear, because the array itself expands by count elements. The to parameter must have a value
between 0 and the size of the array. If it is equal to the size of the array, new elements are added to
the end of the array.
The specified elements are copied from one array to another, i.e., they remain unchanged in the
original array, and their "doubles" in the new array become independent instances that are not related
to the "originals" in any way.
The function returns true if successful or false in case of error.
Let's consider some examples (ArrayInsert.mq5). The OnStart function provides descriptions of several
arrays of different configurations, both fixed and dynamic.

---

## Page 351

Part 4. Common APIs
351 
4.3 Working with arrays
#define PRTS(A) Print(#A, "=", (string)(A) + " / status:" + (string)GetLastError())
void OnStart()
{
   int dynamic[];
   int dynamic2Dx5[][5];
   int dynamic2Dx4[][4];
   int fixed[][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
   int insert[] = {10, 11, 12};
   int array[1] = {100};
   ...
To begin with, for convenience, a macro is introduced that displays the error code (obtained through
the function GetLastError) immediately after calling the instruction under test – PRTS. This is a slightly
modified version of the familiar PRT macro.
Attempts to copy elements between arrays of different configurations end with error 4006
(ERR_INVALID_ARRAY).
   // you can't mix 1D and 2D arrays
   PRTS(ArrayInsert(dynamic, fixed, 0)); // false:4006, ERR_INVALID_ARRAY
   ArrayPrint(dynamic); // empty
   // you can't mix 2D arrays of different configurations by the second dimension
   PRTS(ArrayInsert(dynamic2Dx5, fixed, 0)); // false:4006, ERR_INVALID_ARRAY
   ArrayPrint(dynamic2Dx5); // empty
   // even if both arrays are fixed (or both are dynamic),
   // size by "higher" dimensions must match
   PRTS(ArrayInsert(fixed, insert, 0)); // false:4006, ERR_INVALID_ARRAY
   ArrayPrint(fixed); // not changed   
   ...
The target index must be within the array.
   // target index 10 is out of the range or the array 'insert',
   // could be 0, 1, 2, because its size = 3
   PRTS(ArrayInsert(insert, array, 10)); // false:5052, ERR_SMALL_ARRAY
   ArrayPrint(insert); // not changed   
   ...
The following are successful array modifications:

---

## Page 352

Part 4. Common APIs
352
4.3 Working with arrays
   // copy second row from 'fixed', 'dynamic2Dx4' is allocated
   PRTS(ArrayInsert(dynamic2Dx4, fixed, 0, 1, 1)); // true
   ArrayPrint(dynamic2Dx4);
   // both rows from 'fixed' are added to the end of 'dynamic2Dx4', it expands
   PRTS(ArrayInsert(dynamic2Dx4, fixed, 1)); // true
   ArrayPrint(dynamic2Dx4);
   // memory is allocated for 'dynamic' for all elements 'insert'
   PRTS(ArrayInsert(dynamic, insert, 0)); // true
   ArrayPrint(dynamic);
   // 'dynamic' expands by 1 element
   PRTS(ArrayInsert(dynamic, array, 1)); // true
   ArrayPrint(dynamic);
   // new element will push the last one out of 'insert'
   PRTS(ArrayInsert(insert, array, 1)); // true
   ArrayPrint(insert);
}
Here's what will appear in the log:
ArrayInsert(dynamic2Dx4,fixed,0,1,1)=true
    [,0][,1][,2][,3]
[0,]   5   6   7   8
ArrayInsert(dynamic2Dx4,fixed,1)=true
    [,0][,1][,2][,3]
[0,]   5   6   7   8
[1,]   1   2   3   4
[2,]   5   6   7   8
ArrayInsert(dynamic,insert,0)=true
10 11 12
ArrayInsert(dynamic,array,1)=true
 10 100  11  12
ArrayInsert(insert,array,1)=true
 10 100  11
bool ArrayCopy(void &target[], const void &source[], int to = 0, int from = 0, int count =
WHOLE_ARRAY)
The function copies part or all of the source array to the target array. The place in the target array
where the elements are written is specified by the index in the to parameter. The starting index of the
element from which to start copying from the source array is given by the from index. The
WHOLE_ARRAY constant (-1 ) in the count parameter specifies the transfer of all elements of the source
array. If count is less than zero or greater than the number of elements remaining from the from
position to the end of the source array, the entire remainder of the array is copied.
Unlike the ArrayInsert function, the ArrayCopy function does not shift the existing elements of the
receiving array but writes new elements to the specified positions over the old ones.
All indexes and the number of elements are set taking into account the continuous numbering of
elements, regardless of the number of dimensions in the arrays and their configuration. In other words,
elements can be copied from multidimensional arrays to one-dimensional arrays and vice versa, or

---

## Page 353

Part 4. Common APIs
353
4.3 Working with arrays
between multidimensional arrays with different sizes according to the "higher" dimensions (see the
example).
The function works with fixed and dynamic arrays, as well as time series arrays designated as indicator
buffers.
It is permitted to copy elements from an array to itself. But if the target and source areas overlap, you
need to keep in mind that the iteration is done from left to right.
A dynamic destination array is automatically expanded as needed. Fixed arrays retain their dimensions,
and what is copied must fit in the array, otherwise an error will occur.
Arrays of built-in types and arrays of structures with simple type fields are supported. For numeric
types, the function will try to convert the data if the source and destination types differ. A string array
can only be copied to a string array. Class objects are not allowed, but pointers to objects can be
copied.
The function returns the number of elements copied (0 on error).
In the script ArrayCopy.mq5 there are several examples of using the function.
class Dummy
{
   int x;
};
   
void OnStart()
{
   Dummy objects1[5], objects2[5];
 // error: structures or classes with objects are not allowed
   PRTS(ArrayCopy(objects1, objects2));
   ...
Arrays with objects generate a compilation error stating that "structures or classes containing objects
are not allowed", but pointers can be copied.

---

## Page 354

Part 4. Common APIs
354
4.3 Working with arrays
   Dummy *pointers1[5], *pointers2[5];
   for(int i = 0; i < 5; ++i)
   {
      pointers1[i] = &objects1[i];
   }
   PRTS(ArrayCopy(pointers2, pointers1)); // 5 / status:0
   for(int i = 0; i < 5; ++i)
   {
      Print(i, " ", pointers1[i], " ", pointers2[i]);
   }
 // it outputs some pairwise identical object descriptors
   /*
   0 1048576 1048576
   1 2097152 2097152
   2 3145728 3145728
   3 4194304 4194304
   4 5242880 5242880
   */
Arrays of structures with fields of simple types are also copied without problems.
struct Simple
{
   int x;
};
   
void OnStart()
{
   ...
   Simple s1[3] = {{123}, {456}, {789}}, s2[];
   PRTS(ArrayCopy(s2, s1)); // 3 / status:0
   ArrayPrint(s2);
   /*
       [x]
   [0] 123
   [1] 456
   [2] 789
   */
   ...
To further demonstrate how to work with arrays of different types and configurations, the following
arrays are defined (including fixed, dynamic, and arrays with a different number of dimensions):

---

## Page 355

Part 4. Common APIs
355
4.3 Working with arrays
   int dynamic[];
   int dynamic2Dx5[][5];
   int dynamic2Dx4[][4];
   int fixed[][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
   int insert[] = {10, 11, 12};
   double array[1] = {M_PI};
   string texts[];
   string message[1] = {"ok"};
   ...
When copying one element from the fixed array from position 1  (number 2), a whole row of 4 elements
is allocated in the receiving dynamic array dynamic2Dx4, and since only 1  element is copied, the
remaining three will contain random "garbage" (highlighted in yellow).
   PRTS(ArrayCopy(dynamic2Dx4, fixed, 0, 1, 1)); // 1 / status:0
   ArrayPrint(dynamic2Dx4);
   /*
       [,0][,1][,2][,3]
   [0,]   2   1   2   3
   */
   ...
Next, we copy all the elements from the fixed array, starting from the third one, into the same array
dynamic2Dx4, but starting from position 1 . Since 5 elements are copied (the total number in the array
fixed is 8 minus the initial position 3), and they are placed at index 1 , in total, 1  + 5 will be occupied in
the receiving array, for a total of 6 elements. And since the array dynamic2Dx4 has 4 elements in each
row (in the second dimension), it is possible to allocate memory for it only for the number of elements
that is a multiple of 4, i.e., 2 more elements will be distributed, in which random data will remain.
   PRTS(ArrayCopy(dynamic2Dx4, fixed, 1, 3)); // 5 / status:0
   ArrayPrint(dynamic2Dx4);
   /*
       [,0][,1][,2][,3]
   [0,]   2   4   5   6
   [1,]   7   8   3   4
   */
When copying a multidimensional array to a one-dimensional array, the elements will be presented in a
"flat" form.
   PRTS(ArrayCopy(dynamic, fixed)); // 8 / status:0
   ArrayPrint(dynamic);
   /*
   1 2 3 4 5 6 7 8
   */
When copying a one-dimensional array to a multidimensional one, the elements are "expanded"
according to the dimensions of the receiving array.

---

## Page 356

Part 4. Common APIs
356
4.3 Working with arrays
   PRTS(ArrayCopy(dynamic2Dx5, insert)); // 3 / status:0
   ArrayPrint(dynamic2Dx5);
   /*
       [,0][,1][,2][,3][,4]
   [0,]  10  11  12   4   5
   */
In this case, 3 elements were copied and they fit into one row which is 5 elements long (according to
the configuration of the receiving array). The memory for the remaining two elements of the series was
allocated, but not filled (contains "garbage").
We can overwrite the array dynamic2Dx5 from another source, including from a multidimensional array
of a different configuration. Since two rows of 5 elements each were allocated in the receiving array,
and 2 rows of 4 elements each were allocated in the source array, 2 additional elements were left
unfilled.
   PRTS(ArrayCopy(dynamic2Dx5, fixed)); // 8 / status:0
   ArrayPrint(dynamic2Dx5);
   /*
       [,0][,1][,2][,3][,4]
   [0,]   1   2   3   4   5
   [1,]   6   7   8   0   0
   */
By using ArrayCopy it is possible to change elements in fixed receiver arrays.
   PRTS(ArrayCopy(fixed, insert)); // 3 / status:0
   ArrayPrint(fixed);
   /*
       [,0][,1][,2][,3]
   [0,]  10  11  12   4
   [1,]   5   6   7   8
   */
Here we have overwritten the first three elements of the array fixed. And then let's overwrite the last 3.
   PRTS(ArrayCopy(fixed, insert, 5)); // 3 / status:0
   ArrayPrint(fixed);
   /*
       [,0][,1][,2][,3]
   [0,]  10  11  12   4
   [1,]   5  10  11  12
   */
Copying to a position equal to the length of the fixed array will not work (the dynamic destination array
would expand in this case).
   PRTS(ArrayCopy(fixed, insert, 8)); // 4006, ERR_INVALID_ARRAY
   ArrayPrint(fixed); // no changes
String arrays combined with arrays of other types will throw an error:

---

## Page 357

Part 4. Common APIs
357
4.3 Working with arrays
   PRTS(ArrayCopy(texts, insert)); // 5050, ERR_INCOMPATIBLE_ARRAYS
   ArrayPrint(texts); // empty
But between string arrays, copying is possible:
   PRTS(ArrayCopy(texts, message));
   ArrayPrint(texts); // "ok"
Arrays of different numeric types are copied with the necessary conversion.
   PRTS(ArrayCopy(insert, array, 1)); // 1 / status:0
   ArrayPrint(insert); // 10  3 12
Here we have written the number Pi in an integer array, and therefore received the value 3 (it replaced
1 1 ).
bool ArrayRemove(void &array[], uint start, uint count = WHOLE_ARRAY)
The function removes the specified number of elements from the array starting from the index start. An
array can be multidimensional and have any built-in or structure type with fields of built-in types, with
the exception of strings.
The index start and quantity count refer to the first dimension of the arrays. In other words, for
multidimensional arrays, deletion is performed not by individual elements, but by the entire
configuration described by "higher" dimensions. For example, for a two-dimensional array, the value 1 
in the parameter count means deleting a whole series of length equal to the second dimension (see the
example).
The value start must be between 0 and the size of the first dimension minus 1 .
The function cannot be applied to arrays with time series (built-in timeseries or indicator buffers).
To test the function, we prepared the script ArrayRemove.mq5. In particular, it defines 2 structures:
struct Simple
{
   int x;
};
   
struct NotSoSimple
{
   int x;
   string s; // a field of type string causes the compiler to make an implicit destructor
};
Arrays with a simple structure can be processed by the function ArrayRemove successfully, while
arrays of objects with destructors (even with implicit ones, as in NotSoSimple) cause an error:

---

## Page 358

Part 4. Common APIs
358
4.3 Working with arrays
void OnStart()
{
   Simple structs1[10];
   PRTS(ArrayRemove(structs1, 0, 5)); // true / status:0
   
   NotSoSimple structs2[10];
   PRTS(ArrayRemove(structs2, 0, 5)); // false / status:4005,
                                      // ERR_STRUCT_WITHOBJECTS_ORCLASS
   ...
Next, arrays of various configurations are defined and initialized.
   int dynamic[];
   int dynamic2Dx4[][4];
   int fixed[][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
   
   // make 2 copies
   ArrayCopy(dynamic, fixed);
   ArrayCopy(dynamic2Dx4, fixed);
   
   // show initial data
   ArrayPrint(dynamic);
   /*
   1 2 3 4 5 6 7 8
   */
   ArrayPrint(dynamic2Dx4);
   /*
       [,0][,1][,2][,3]
   [0,]   1   2   3   4
   [1,]   5   6   7   8
   */
When deleting from a fixed array, all elements after the fragment being removed are shifted to the left.
It is important that the size of the array does not change, and therefore copies of the shifted elements
appear in duplicate.
   PRTS(ArrayRemove(fixed, 0, 1));
   ArrayPrint(fixed);
   /*
   ArrayRemove(fixed,0,1)=true / status:0
       [,0][,1][,2][,3]
   [0,]   5   6   7   8
   [1,]   5   6   7   8
   */
Here we removed one element of the first dimension of a two-dimensional array fixed by offset 0, that
is, the initial row. The elements of the next row moved up and remained in the same row.
If we perform the same operation with a dynamic array (identical in content to the array fixed), its size
will be automatically reduced by the number of elements removed.

---

## Page 359

Part 4. Common APIs
359
4.3 Working with arrays
   PRTS(ArrayRemove(dynamic2Dx4, 0, 1));
   ArrayPrint(dynamic2Dx4);
   /*
   ArrayRemove(dynamic2Dx4,0,1)=true / status:0
       [,0][,1][,2][,3]
   [0,]   5   6   7   8
   */
In a one-dimensional array, each element removed corresponds to a single value. For example, in the
array dynamic, when removing three elements starting at index 2, we get the following result:
   PRTS(ArrayRemove(dynamic, 2, 3));
   ArrayPrint(dynamic);
   /*
   ArrayRemove(dynamic,2,3)=true / status:0
   1 2 6 7 8
   */
The values 3, 4, 5 have been removed, the array size has been reduced by 3.
bool ArrayReverse(void &array[], uint start = 0, uint count = WHOLE_ARRAY)
The function reverses the order of the specified elements in the array. Elements to be reversed are
determined by a starting position start and quantity count. If start = 0, and count = WHOLE_ ARRAY, the
entire array is accessed.
Arrays of arbitrary dimensions and types are supported, both fixed and dynamic (including time series
in indicator buffers). An array can contain objects, pointers, or structures. For multidimensional arrays,
only the first dimension is reversed.
The count value must be between 0 and the number of elements in the first dimension. Please note that
count less than 2 will not give a noticeable effect, but it can be used to unify loops in algorithms.
The function returns true if successful or false in case of error.
The ArrayReverse.mq5 script can be used to test the function. At its beginning, a class is defined for
generating objects stored in an array. The presence of strings and other "complex" fields is not a
problem.
class Dummy
{
   static int counter;
   int x;
   string s; // a field of type string causes the compiler to create an implicit destructor
public:
   Dummy() { x = counter++; }
};
   
static int Dummy::counter;
Objects are identified by a serial number (assigned at the time of creation).

---

## Page 360

Part 4. Common APIs
360
4.3 Working with arrays
void OnStart()
{
   Dummy objects[5];
   Print("Objects before reverse");
   ArrayPrint(objects);
   /*
       [x]  [s]
   [0]   0 null
   [1]   1 null
   [2]   2 null
   [3]   3 null
   [4]   4 null
   */
After applying ArrayReverse we get the expected reverse order of the objects.
   PRTS(ArrayReverse(objects)); // true / status:0
   Print("Objects after reverse");
   ArrayPrint(objects);
   /*
       [x]  [s]
   [0]   4 null
   [1]   3 null
   [2]   2 null
   [3]   1 null
   [4]   0 null
   */
Next, numerical arrays of different configurations are prepared and unfolded with different parameters.

---

## Page 361

Part 4. Common APIs
361 
4.3 Working with arrays
   int dynamic[];
   int dynamic2Dx4[][4];
   int fixed[][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
   
   ArrayCopy(dynamic, fixed);
   ArrayCopy(dynamic2Dx4, fixed);
   
   PRTS(ArrayReverse(fixed)); // true / status:0
   ArrayPrint(fixed);
   /*
       [,0][,1][,2][,3]
   [0,]   5   6   7   8
   [1,]   1   2   3   4
   */
   
   PRTS(ArrayReverse(dynamic, 4, 3)); // true / status:0
   ArrayPrint(dynamic);
   /*
   1 2 3 4 7 6 5 8
   */
   
   PRTS(ArrayReverse(dynamic, 0, 1)); // does nothing (count = 1)
   PRTS(ArrayReverse(dynamic2Dx4, 2, 1)); // false / status:5052, ERR_SMALL_ARRAY
}
In the latter case, the value start (2) exceeds the size in the first dimension, so an error occurs.
4.3.6 Moving (swapping) arrays
MQL5 provides the ability to swap the contents of two arrays in a resource-efficient way (without
physical allocation of memory and copying data). In some other programming languages, a similar
operation is supported not only for arrays, but also for other variables, and is called moving.
bool ArraySwap(void &array1 [], void &array2[])
The function swaps the contents of two dynamic arrays of the same type. Arrays of any type are
supported. However, the function is not applicable to timeseries arrays and indicator buffers, as well as
to any arrays with the const modifier.
For multidimensional arrays, the number of elements in all dimensions except the first must match.
The function returns true if successful or false in case of error.
The main use of the function is to speed up the program by eliminating the physical copying of the
array when it is passed to or returned from the function, and it is known that the source array is no
longer needed. The fact is that swapping takes place almost instantly since the application data does
not move in any way. Instead, there is an exchange of meta-data about arrays stored in service
structures that describe dynamic arrays (and this takes only 52 bytes).
Suppose there is a class intended for processing an array by certain algorithms. The same array can be
subjected to different operations and therefore it makes sense to keep it as a class member. But then
there is a question, how to transfer it to an object? In MQL5, methods (as well as functions in general)
allow passing arrays only by reference. Putting aside all kinds of wrapper classes that contain an array

---

## Page 362

Part 4. Common APIs
362
4.3 Working with arrays
and are passed by pointer, the only simple solution seems to be the following: to describe, for example,
an array parameter in the class constructor and copy it to the internal array using ArrayCopy. But it is
more efficient to use ArraySwap.
template<typename T>
class Worker
{
   T array[];
   
public:
   Worker(T &source[])
   {
      // ArrayCopy(array, source); // memory and time consuming 
      ArraySwap(source, array);
   }
   ...
};
Since the array array was empty before the swap, after the operation the array used as the source
argument will become empty, while array will be filled with input data with little or no overhead.
After the object of the class becomes the "owner" of the array, we can modify it with the required
algorithms, for example, through a special method process, which takes the code of the requested
algorithm as a parameter. It can be sorting, smoothing, mixing, adding noise and much more. But first,
let's try to test the idea on a simple operation of array reversal by the function ArrayReverse (see file
ArraySwapSimple.mq5).
   bool process(const int mode)
   {
      if(ArraySize(array) == 0) return false;
      switch(mode)
      {
      case -4:
         // example: shuffling
         break;
      case -3:
         // example: logarithm
         break;
      case -2:
         // example: adding noise
         break;
      case -1:
         ArrayReverse(array);
         break;
      ...
      }
      return true;
   }
You can provide access to the results of work using two methods: element by element (by overloading
the '[]' operator) or by an entire array (again we use ArraySwap in the corresponding method get, but
you can also provide a method for copying through ArrayCopy).

---

## Page 363

Part 4. Common APIs
363
4.3 Working with arrays
   T operator[](int i)
   {
      return array[i];
   }
   
   void get(T &result[])
   {
      ArraySwap(array, result);
   }
For the purpose of universality, the class is made template. This will allow adapting it in the future for
arrays of arbitrary structures, but for now, you can check the inversion of a simple array of the type
double:
void OnStart()
{
   double data[];
   ArrayResize(data, 3);
   data[0] = 1;
   data[1] = 2;
   data[2] = 3;
   
   PRT(ArraySize(data));        // 3
   Worker<double> simple(data);
   PRT(ArraySize(data));        // 0
   simple.process(-1);  // reversing array
   
   double res[];
   simple.get(res);
   ArrayPrint(res); // 3.00000 2.00000 1.00000
}
The task of sorting is more realistic, and for an array of structures, sorting by any field may be
required. In the next section we will study in detail the function ArraySort, which allows you to sort in
ascending order an array of any built-in type, but not structures. There we will try to eliminate this
"gap", leaving ArraySwap in action.
4.3.7 Comparing, sorting, and searching in arrays
The MQL5 API contains several functions that allow comparing and sorting arrays, as well as searching
for the maximum, minimum, or any specific value in them.
int ArrayCompare(const void &array1 [], const void &array2[], int start1  = 0, int start2 = 0, int count
= WHOLE_ARRAY)
The function returns the result of comparing two arrays of built-in types or structures with fields of
built-in types, excluding strings. Arrays of class objects are not supported. Also, you cannot compare
arrays of structures that contain dynamic arrays, class objects, or pointers.
By default, the comparison is performed for entire arrays but, if necessary, you can specify parts of
arrays, for which there are parameters start1  (starting position in the first array), start2 (starting
position in the second array), and count.

---

## Page 364

Part 4. Common APIs
364
4.3 Working with arrays
Arrays can be fixed or dynamic, as well as multidimensional. During comparison, multidimensional
arrays are represented as equivalent one-dimensional arrays (for example, for two-dimensional arrays,
the elements of the second row follow the elements of the first, the elements of the third row follow the
second, and so on). For this reason, the parameters start1 , start2, and count for multidimensional
arrays are specified through element numbering, and not an index along the first dimension.
Using various start1  and start2 offsets you can compare different parts of the same array.
Arrays are compared element by element until the first discrepancy is found or the end of one of the
arrays is reached. The relationship between two elements (which are in the same positions in both
arrays) depends on the type: for numbers, the operators '>', '<', '==' are used, and for strings, the
StringCompare function is used. Structures are compared byte by byte, which is equivalent to
executing the following code for each pair of elements:
uchar bytes1[], bytes2[];
StructToCharArray(array1[i], bytes1);
StructToCharArray(array2[i], bytes2);
int cmp = ArrayCompare(bytes1, bytes2);
Based on the ratio of the first differing elements, the result of bulk comparison of the arrays array1  and
array2 is obtained. If no differences are found, and the length of the arrays is equal, then the arrays are
considered the same. If the length is different, then the longer array is considered greater.
The function returns -1  if array1  is "less than" array2, +1  if array1  is "greater than" array2, and 0 if
they are "equal".
In case of an error, the result is -2.
Let's look at some examples in the script ArrayCompare.mq5.
Let's create a simple structure for filling the arrays to be compared.
struct Dummy
{
   int x;
   int y;
   
   Dummy()
   {
      x = rand() / 10000;
      y = rand() / 5000;
   }
};
The class fields are filled with random numbers (each time the script is run, we will receive new values).
In the OnStart function, we describe a small array of structures and compare successive elements with
each other (as moving neighboring fragments of an array with the length of 1  element).

---

## Page 365

Part 4. Common APIs
365
4.3 Working with arrays
#define LIMIT 10
void OnStart()
{
   Dummy a1[LIMIT];
   ArrayPrint(a1);
   
   // pairwise comparison of neighboring elements
   // -1: [i] < [i + 1]
   // +1: [i] > [i + 1]
   for(int i = 0; i < LIMIT - 1; ++i)
   {
      PRT(ArrayCompare(a1, a1, i, i + 1, 1));
   }
   ...
Below are the results for one of the array options (for the convenience of analysis, the column with the
signs "greater than" (+1 ) / "less than" (-1 ) is added directly to the right of the contents of the array):
       [x] [y]   // result
   [0]   0   3   // -1
   [1]   2   4   // +1
   [2]   2   3   // +1
   [3]   1   6   // +1
   [4]   0   6   // -1
   [5]   2   0   // +1
   [6]   0   4   // -1
   [7]   2   5   // +1
   [8]   0   5   // -1
   [9]   3   6
Comparing the two halves of the array to each other gives -1 :
 // compare first and second half
   PRT(ArrayCompare(a1, a1, 0, LIMIT / 2, LIMIT / 2)); // -1
Next, we will compare arrays of strings with predefined data.
   string s[] = {"abc", "456", "$"};
   string s0[][3] = {{"abc", "456", "$"}};
   string s1[][3] = {{"abc", "456", ""}};
   string s2[][3] = {{"abc", "456"}}; // last element omitted: it is null
   string s3[][2] = {{"abc", "456"}};
   string s4[][2] = {{"aBc", "456"}};
   
   PRT(ArrayCompare(s0, s));  // s0 == s, 1D and 2D arrays contain the same data
   PRT(ArrayCompare(s0, s1)); // s0 > s1 since "$" > ""
   PRT(ArrayCompare(s1, s2)); // s1 > s2 since "" > null
   PRT(ArrayCompare(s2, s3)); // s2 > s3 due to different lengths: [3] > [2]
   PRT(ArrayCompare(s3, s4)); // s3 < s4 since "abc" < "aBc"
Finally, let's check the ratio of array fragments:

---

## Page 366

Part 4. Common APIs
366
4.3 Working with arrays
   PRT(ArrayCompare(s0, s1, 1, 1, 1)); // second elements (with index 1) are equal 
   PRT(ArrayCompare(s1, s2, 0, 0, 2)); // the first two elements are equal
bool ArraySort(void &array[])
The function sorts a numeric array (including possibly a multidimensional array) by the first dimension.
The sorting order is ascending. To sort an array in descending order, apply the ArrayReverse function
to the resulting array or process it in reverse order.
The function does not support arrays of strings, structures, or classes.
The function returns true if successful or false in case of error.
If the "timeseries" property is set for an array, then the elements in it are indexed in the reverse order
(see details in section Array indexing direction as in timeseries), and this has an "external" reversal
effect on the sorting order: when you process such an array directly, you will get descending values. At
the physical level, the array is always sorted in ascending order, and that is how it is stored.
In the script ArraySort.mq5 a 1 0 by 3, 2-dimensional array is generated and sorted using ArraySort:
#define LIMIT 10
#define SUBLIMIT 3
   
void OnStart()
{
   // generating random data
   int array[][SUBLIMIT];
   ArrayResize(array, LIMIT);
   for(int i = 0; i < LIMIT; ++i)
   {
      for(int j = 0; j < SUBLIMIT; ++j)
      {
         array[i][j] = rand();
      }
   }
   
   Print("Before sort");
   ArrayPrint(array);    // source array
   
   PRTS(ArraySort(array));
   
   Print("After sort");
   ArrayPrint(array);    // ordered array
   ...
}
According to the log, the first column is sorted in ascending order (specific numbers will vary due to
random generation):

---

## Page 367

Part 4. Common APIs
367
4.3 Working with arrays
Before sort
      [,0]  [,1]  [,2]
[0,]  8955  2836 20011
[1,]  2860  6153 25032
[2,] 16314  4036 20406
[3,] 30366 10462 19364
[4,] 27506  5527 21671
[5,]  4207  7649 28701
[6,]  4838   638 32392
[7,] 29158 18824 13536
[8,] 17869 23835 12323
[9,] 18079  1310 29114
ArraySort(array)=true / status:0
After sort
      [,0]  [,1]  [,2]
[0,]  2860  6153 25032
[1,]  4207  7649 28701
[2,]  4838   638 32392
[3,]  8955  2836 20011
[4,] 16314  4036 20406
[5,] 17869 23835 12323
[6,] 18079  1310 29114
[7,] 27506  5527 21671
[8,] 29158 18824 13536
[9,] 30366 10462 19364
The values in the following columns have moved synchronously with the "leading" values in the first
column. In other words, the entire rows are permuted, despite the fact that only the first column is the
sorting criterion.
But what if you want to sort a two-dimensional array by a column other than the first one? You can
write a special algorithm for that. One of the options is included in the file ArraySort.mq5 as a template
function:

---

## Page 368

Part 4. Common APIs
368
4.3 Working with arrays
template<typename T>
bool ArraySort(T &array[][], const int column)
{
   if(!ArrayIsDynamic(array)) return false;
   
   if(column == 0)
   {
      return ArraySort(array); // standard function 
   }
   
   const int n = ArrayRange(array, 0);
   const int m = ArrayRange(array, 1);
   
   T temp[][2];
   
   ArrayResize(temp, n);
   for(int i = 0; i < n; ++i)
   {
      temp[i][0] = array[i][column];
      temp[i][1] = i;
   }
   
   if(!ArraySort(temp)) return false;
   
   ArrayResize(array, n * 2);
   for(int i = n; i < n * 2; ++i)
   {
      ArrayCopy(array, array, i * m, (int)(temp[i - n][1] + 0.1) * m, m);
      /* equivalent
      for(int j = 0; j < m; ++j)
      {
         array[i][j] = array[(int)(temp[i - n][1] + 0.1)][j];
      }
      */
   }
   
   return ArrayRemove(array, 0, n);
}
The given function only works with dynamic arrays because the size of array is doubled to assemble
intermediate results in the second half of the array, and finally, the first half (original) is removed with
ArrayRemove. That is why the original test array in the OnStart function was distributed through
ArrayResize.
We encourage you to study the sorting principle on your own (or turn over a couple of pages).
Something similar should be implemented for arrays with a large number of dimensions (for example,
array[][][]).
Now recall that in the previous section, we raised the issue of sorting an array of structures by an
arbitrary field. As we know, the standard ArraySort function is not able to do this. Let's try to come up

---

## Page 369

Part 4. Common APIs
369
4.3 Working with arrays
with a "bypass route". Let's take the class from the ArraySwapSimple.mq5 file from the previous
section as a basis. Let's copy it to ArrayWorker.mq5 and add the required code.
In the Worker::process method, we will provide a call to the auxiliary sorting method arrayStructSort,
and the field to be sorted will be specified by number (how it can be done, we will describe below):
   ...
   bool process(const int mode)
   {
      ...
      switch(mode)
      {
      ...
      case -1:
         ArrayReverse(array);
         break;
      default: // sorting by field number 'mode'
         arrayStructSort(mode);
         break;
      }
      return true;
   }
   
private:
   bool arrayStructSort(const int field)
   {
      ...
   }
Now it becomes clear why all the previous modes (values of the mode parameter) in the process
method were negative: zero and positive values are reserved for sorting and correspond to the
"column" number.
The idea of sorting an array of structures is taken from sorting a two-dimensional array. We only need
to somehow map a single structure to a one-dimensional array (representing a row of a two-
dimensional array). To do this, firstly, you need to decide what type the array should be.
Since the worker class is already a template, we will add one more parameter to the template so that
the array type can be flexibly set.
template<typename T, typename R>
class Worker
{
   T array[];
   ...
Now, let's get back to associations, which allow you to overlay variables of different types on top of
each other. Thus, we get the following tricky construction:

---

## Page 370

Part 4. Common APIs
370
4.3 Working with arrays
   union Overlay
   {
      T r;
      R d[sizeof(T) / sizeof(R)];
   };
In this union, the type of the structure is combined with an array of type R, and its size is automatically
calculated by the compiler based on the ratio of the sizes of two types, T and R.
Now, inside the arrayStructSort method, we can partially duplicate the code of two-dimensional array
sorting.
   bool arrayStructSort(const int field)
   {
      const int n = ArraySize(array);
      
      R temp[][2];
      Overlay overlay;
      
      ArrayResize(temp, n);
      for(int i = 0; i < n; ++i)
      {
         overlay.r = array[i];
         temp[i][0] = overlay.d[field];
         temp[i][1] = i;
      }
      ...
Instead of an array with the original structures, we prepare the temp[][2] array of type R, extend it to
the number of records in array, and write the following in the loop: the "display" of the required field
field from the structure at the 0th index of each row, and the original index of this element at the 1 st
index.
The "display" is based on the fact that fields in structures are usually aligned in some way since they
use standard types. Therefore, with a properly chosen R type, it is possible to provide full or partial
hitting of fields in the array elements in the "overlay".
For example, in the standard structure MqlRates the first 6 fields are 8 bytes in size, and therefore map
correctly onto the array double or long (these are R template type candidates).
struct MqlRates 
{ 
   datetime time; 
   double   open; 
   double   high; 
   double   low; 
   double   close; 
   long     tick_volume; 
   int      spread; 
   long     real_volume; 
};
With the last two fields, the situation is more complicated. If the field spread still can be reached using
type int as R, then the field real_ volume turns out to be at an offset that is not a multiple of its own size

---

## Page 371

Part 4. Common APIs
371 
4.3 Working with arrays
(due to the field type int, i.e. 4 bytes, before it). These are problems of a particular method. It can be
improved, or another method can be invented.
But let's go back to the sorting algorithm. After the array temp is populated, it can be sorted with the
usual function ArraySort, and then the original indexes can be used to form a new array with the
correct structure order.
      ...
      if(!ArraySort(temp)) return false;
      T result[];
      
      ArrayResize(result, n);
      for(int i = 0; i < n; ++i)
      {
         result[i] = array[(int)(temp[i][1] + 0.1)];
      }
      
      return ArraySwap(result, array);
   }
Before exiting the function, we use ArraySwap again, in order to replace the contents of an intra-object
array array in a resource-efficient way with something new and ordered, which is received in the local
array result.
Let's check the class worker in action: in the function OnStart let's define an array of structures
MqlRates and ask the terminal for several thousand records.
#define LIMIT 5000
void OnStart()
{
   MqlRates rates[];
   int n = CopyRates(_Symbol, _Period, 0, LIMIT, rates);
   ...
The CopyRates function will be described in a separate section. For now, it's enough for us to know that
it fills the passed array rates with quotes of the symbol and timeframe of the current chart on which
the script is running. The macro LIMIT specifies the number of requested bars: you need to make sure
that this value is not greater than your terminal's setting for the number of bars in each window.
To process the received data, we will create an object worker with types T=MqlRates and R=double:
Worker<MqlRates, double> worker(rates);
Sorting can be started with an instruction of the following form:
worker.process(offsetof(MqlRates, open) / sizeof(double));
Here we use the offsetof operator to get the byte offset of the field open inside the structure. It is
further divided by the size double and gives the correct "column" number for sorting by the open price.
You can read the sorting result element by element, or get the entire array:

---

## Page 372

Part 4. Common APIs
372
4.3 Working with arrays
Print(worker[i].open);
...
worker.get(rates);
ArrayPrint(rates);
Note that getting an array by the method get moves it out of the inner array array to the outer one
(passed as an argument) with ArraySwap. So, after that the calls worker.process() are pointless: there
is no more data in the object worker.
To simplify the start of sorting by different fields, an auxiliary function sort has been implemented:
void sort(Worker<MqlRates, double> &worker, const int offset, const string title)
{
   Print(title);
   worker.process(offset);
   Print("First struct");
   StructPrint(worker[0]);
   Print("Last struct");
   StructPrint(worker[worker.size() - 1]);
}
It outputs a header and the first and last elements of the sorted array to the log. With its help, testing
in OnStart for three fields looks like this:
void OnStart()
{
   ...
   Worker<MqlRates, double> worker(rates);
   sort(worker, offsetof(MqlRates, open) / sizeof(double), "Sorting by open price...");
   sort(worker, offsetof(MqlRates, tick_volume) / sizeof(double), "Sorting by tick volume...");
   sort(worker, offsetof(MqlRates, time) / sizeof(double), "Sorting by time...");
}
Unfortunately, the standard function print does not support printing of single structures, and there is no
built-in function StructPrint in MQL5. Therefore, we had to write it ourselves, based on ArrayPrint: in
fact, it is enough to put the structure in an array of size 1 .
template<typename S>
void StructPrint(const S &s)
{
   S temp[1];
   temp[0] = s;
   ArrayPrint(temp);
}
As a result of running the script, we can get something like the following (depending on the terminal
settings, namely on which symbol/timeframe it is executed):

---

## Page 373

Part 4. Common APIs
373
4.3 Working with arrays
Sorting by open price...
First struct
                 [time]  [open]  [high]   [low] [close] [tick_volume] [spread] [real_volume]
[0] 2021.07.21 10:30:00 1.17557 1.17584 1.17519 1.17561          1073        0             0
Last struct
                 [time]  [open]  [high]   [low] [close] [tick_volume] [spread] [real_volume]
[0] 2021.05.25 15:15:00 1.22641 1.22664 1.22592 1.22618           852        0             0
Sorting by tick volume...
First struct
                 [time]  [open]  [high]   [low] [close] [tick_volume] [spread] [real_volume]
[0] 2021.05.24 00:00:00 1.21776 1.21811 1.21764 1.21794            52       20             0
Last struct
                 [time]  [open]  [high]   [low] [close] [tick_volume] [spread] [real_volume]
[0] 2021.06.16 21:30:00 1.20436 1.20489 1.20149 1.20154          4817        0             0
Sorting by time...
First struct
                 [time]  [open]  [high]   [low] [close] [tick_volume] [spread] [real_volume]
[0] 2021.05.14 16:15:00 1.21305 1.21411 1.21289 1.21333           888        0             0
Last struct
                 [time]  [open]  [high]   [low] [close] [tick_volume] [spread] [real_volume]
[0] 2021.07.27 22:45:00 1.18197 1.18227 1.18191 1.18225           382        0             0
Here is the data for EURUSD,M1 5.
The above implementation of sorting is potentially one of the fastest because it uses the built-in
ArraySort.
If, however, the difficulties with aligning the fields of the structure or the skepticism towards the very
approach of "mapping" the structure into an array force us to abandon this method (and thus, the
function ArraySort), the proven "do-it-yourself" method remains at our disposal.
There are a large number of sorting algorithms that are easy to adapt to MQL5. One of the quick
sorting options is presented in the file QuickSortStructT.mqh attached to the book. This is an improved
version QuickSortT.mqh, which we used in the section String comparison. It has the method Compare of
the template class QuickSortStructT which is made purely virtual and must be redefined in the
descendant class to return an analog of the comparison operator '>' for the required type and its fields.
For the user convenience, a macro has been created in the header file:
#define SORT_STRUCT(T, A, F)                                           \
{                                                                    \
   class InternalSort : public QuickSortStructT<T>                   \
   {                                                                 \
      virtual bool Compare(const T &a, const T &b) override          \
      {                                                              \
         return a.##F > b.##F;                                       \
      }                                                              \
   } sort;                                                           \
   sort.QuickSort(A);                                                \
}
Using it, to sort an array of structures by a given field, it is enough to write one instruction. For
example:

---

## Page 374

Part 4. Common APIs
374
4.3 Working with arrays
   MqlRates rates[];
   CopyRates(_Symbol, _Period, 0, 10000, rates);
   SORT_STRUCT(MqlRates, rates, high);
Here the rates array of type MqlRates is sorted by the high price.
int ArrayBsearch(const type &array[], type value)
The function searches a given value in a numeric array. Arrays of all built-in numeric types are
supported. The array must be sorted in ascending order by the first dimension, otherwise the result will
be incorrect.
The function returns the index of the matching element (if there are several, then the index of the first
of them) or the index of the element closest in value (if there is no exact match), ti.e., it can be an
element with either a larger or smaller value than the one being searched for. If the desired value is
less than the first (minimum), then 0 is returned. If the searched value is greater than the last
(maximum), its index is returned.
The index depends on the direction of the numbering of the elements in the array: direct (from the
beginning to the end) or reverse (from the end to the beginning). It can be recognized and changed
using the functions described in the section Array indexing direction as in timeseries.
If an error occurs, -1  is returned.
For multidimensional arrays, the search is limited to the first dimension.
In the script ArraySearch.mq5 one can find examples of using the function ArrayBsearch.
void OnStart()
{
   int array[] = {1, 5, 11, 17, 23, 23, 37};
     // indexes  0  1   2   3   4   5   6
   int data[][2] = {{1, 3}, {3, 2}, {5, 10}, {14, 10}, {21, 8}};
     // indexes     0       1       2         3         4
   int empty[];
   ...
For three predefined arrays (one of them is empty), the following statements are executed:

---

## Page 375

Part 4. Common APIs
375
4.3 Working with arrays
   PRTS(ArrayBsearch(array, -1)); // 0
   PRTS(ArrayBsearch(array, 11)); // 2
   PRTS(ArrayBsearch(array, 12)); // 2
   PRTS(ArrayBsearch(array, 15)); // 3
   PRTS(ArrayBsearch(array, 23)); // 4
   PRTS(ArrayBsearch(array, 50)); // 6
   
   PRTS(ArrayBsearch(data, 7));   // 2
   PRTS(ArrayBsearch(data, 9));   // 2
   PRTS(ArrayBsearch(data, 10));  // 3
   PRTS(ArrayBsearch(data, 11));  // 3
   PRTS(ArrayBsearch(data, 14));  // 3
   
   PRTS(ArrayBsearch(empty, 0));  // -1, 5053, ERR_ZEROSIZE_ARRAY
   ...
Further, in the populateSortedArray helper function, the numbers array is filled with random values, and
the array is constantly maintained in a sorted state using ArrayBsearch.

---

## Page 376

Part 4. Common APIs
376
4.3 Working with arrays
void populateSortedArray(const int limit)
{
   double numbers[];  // array to fill
 doubleelement[1];// new value to insert
   
   ArrayResize(numbers, 0, limit); // allocate memory beforehand
   
   for(int i = 0; i < limit; ++i)
   {
      // generate a random number
      element[0] = NormalizeDouble(rand() * 1.0 / 32767, 3);
      // find where its place in the array
      int cursor = ArrayBsearch(numbers, element[0]);
      if(cursor == -1)
      {
         if(_LastError == 5053) // empty array
         {
            ArrayInsert(numbers, element, 0);
         }
         else break; // error
      }
      else
      if(numbers[cursor] > element[0]) // insert at 'cursor' position 
      {
         ArrayInsert(numbers, element, cursor);
      }
      else // (numbers[cursor] <= value) // insert after 'cursor'
      {
         ArrayInsert(numbers, element, cursor + 1);
      }
   }
   ArrayPrint(numbers, 3);
}
Each new value goes first into a one-element array element, because this way it's easier to insert it
into the resulting array numbers using the function ArrayInsert.
ArrayBsearch allows you to determine where the new value should be inserted.
The result of the function is displayed in the log:

---

## Page 377

Part 4. Common APIs
377
4.3 Working with arrays
void OnStart()
{
   ...
   populateSortedArray(80);
   /*
    example (will be different on each run due to randomization)
   [ 0] 0.050 0.065 0.071 0.106 0.119 0.131 0.145 0.148 0.154 0.159
        0.184 0.185 0.200 0.204 0.213 0.216 0.220 0.224 0.236 0.238
   [20] 0.244 0.259 0.267 0.274 0.282 0.293 0.313 0.334 0.346 0.366
        0.386 0.431 0.449 0.461 0.465 0.468 0.520 0.533 0.536 0.541
   [40] 0.597 0.600 0.607 0.612 0.613 0.617 0.621 0.623 0.631 0.634
        0.646 0.658 0.662 0.664 0.670 0.670 0.675 0.686 0.693 0.694
   [60] 0.725 0.739 0.759 0.762 0.768 0.783 0.791 0.791 0.791 0.799
        0.838 0.850 0.854 0.874 0.897 0.912 0.920 0.934 0.944 0.992
   */
int ArrayMaximum(const type &array[], int start = 0, int count = WHOLE_ARRAY)
int ArrayMinimum(const type &array[], int start = 0, int count = WHOLE_ARRAY)
The functions ArrayMaximum and ArrayMinimum search a numeric array for the elements with the
maximum and minimum values, respectively. The range of indexes for searching is set by start and
count parameters: with default values, the entire array is searched.
The function returns the position of the found element.
If the "serial" property ("timeseries") is set for an array, the indexing of elements in it is carried out in
the reverse order, and this affects the result of this function (see the example). Built-in functions for
working with the "serial" property are discussed in the next section. More details about "serial" arrays
will be discussed in the chapters on timeseries and indicators.
In multidimensional arrays, the search is performed on the first dimension.
If there are several identical elements in the array with a maximum or minimum value, the function will
return the index of the first of them.
An example of using functions is given in the file ArrayMaxMin.mq5.

---

## Page 378

Part 4. Common APIs
378
4.3 Working with arrays
#define LIMIT 10
   
void OnStart()
{
   // generating random data
   int array[];
   ArrayResize(array, LIMIT);
   for(int i = 0; i < LIMIT; ++i)
   {
      array[i] = rand();
   }
   
   ArrayPrint(array);
   // by default, the new array is not a timeseries
   PRTS(ArrayMaximum(array));
   PRTS(ArrayMinimum(array));
   // turn on the "serial" property
   PRTS(ArraySetAsSeries(array, true));
   PRTS(ArrayMaximum(array));
   PRTS(ArrayMinimum(array));
}
The script will log something like the following set of strings (due to random data generation, each run
will be different):
22242 5909 21570 5850 18026 24740 10852 2631 24549 14635
ArrayMaximum(array)=5 / status:0
ArrayMinimum(array)=7 / status:0
ArraySetAsSeries(array,true)=true / status:0
ArrayMaximum(array)=4 / status:0
ArrayMinimum(array)=2 / status:0
4.3.8 Timeseries indexing direction in arrays
Due to the applied trading specifics, MQL5 brings additional features to working with arrays. One of
them is that array elements can contain data corresponding to time points. These include for example,
arrays with financial instrument quotes, price ticks, and readings of technical indicators. The
chronological order of the data means that new elements are constantly added to the end of the array
and their indexes increase.
However, from the point of view of trading, it is more convenient to count from the present to the past.
Then element 0 always contains the most recent, up-to-date value, element 1  always contains the
previous value, and so on.
MQL5 allows you to select and switch the direction of array indexing on the go. An array numbered
from the present to the past is called a timeseries. If the indexing increase occurs from the past to the
present, this is a regular array. In timeseries, the time decreases with the growth of indices. In
ordinary arrays, the time increases, as in real life.
It is important to note that an array does not have to contain time-related values in order to be able to
switch the addressing order for it. It's just that this feature is most in demand and, in fact, appeared to
work with historical data.

---

## Page 379

Part 4. Common APIs
379
4.3 Working with arrays
This array attribute does not affect the layout of data in memory. Only the order of numbering
changes. In particular, we could implement its analogue in MQL5 ourselves by traversing the array in a
"back to front" loop. But MQL5 provides ready-made functions to hide all this routine from application
programmers.
Timeseries can be any one-dimensional dynamic array described in an MQL program, as well as
external arrays passed to the MQL program from the MetaTrader 5 core, such as parameters of utility
functions. For example, a special type of MQL programs, indicators receives arrays with price data of
the current chart in the OnCalculate event handler. We will study all the features of the applied use of
timeseries later, in the fifth Part of the book.
Arrays defined in an MQL program are not timeseries by default.
Let's consider a set of functions for determining and changing the "series" attribute of an array, as well
as its "belonging" to the terminal. The general ArrayAsSeries.mq5 script with examples will be given
after the description.
bool ArrayIsSeries(const void &array[])
The function returns a sign of whether the specified array is a "real" timeseries, i.e., it is controlled and
provided by the terminal itself. You cannot change this characteristic of an array. Such arrays are
available to the MQL program in the "read-only" mode.
In the MQL5 documentation, the terms "timeseries" and "series" are used to describe both the
reverse indexing of an array and the fact that the array can "belong" to the terminal (the terminal
allocates memory for it and fills it with data). In the book, we will try to avoid this ambiguity and
refer to arrays with reverse indexing as "timeseries". And the terminal arrays will be just terminal's
own arrays.
You can change the indexing of any custom array of the terminal at your discretion by switching it to
the timeseries mode or back to the standard one. This is done using the function ArraySetAsSeries,
which is applicable not only to own, but also to custom dynamic arrays (see below).
bool ArrayGetAsSeries(const void &array[])
The function returns a sign of whether the timeseries indexing mode is enabled for the specified array,
that is, indexing increases in the direction from the present to the past. You can change the indexing
direction using the ArraySetAsSeries function.
The direction of indexing affects values returned by the functions ArrayBsearch, ArrayMaximum, and
ArrayMinimum (see section Comparing, sorting and searching in arrays).
bool ArraySetAsSeries(const void &array[], bool as_series)
The function sets the indexing direction in the array according to the as_ series parameter: the true
value means the reverse order of indexing, while false means the normal order of elements.
The function returns true on successful attribute setting, or false in case of an error.
Arrays of any type are supported, but changing the direction of indexing is prohibited for
multidimensional and fixed-size arrays.
The ArrayAsSeries.mq5 script describes several small arrays for experiments involving the above
functions.

---

## Page 380

Part 4. Common APIs
380
4.3 Working with arrays
#define LIMIT 10
template<typename T>
void indexArray(T &array[])
{
   for(int i = 0; i < ArraySize(array); ++i)
   {
      array[i] = (T)(i + 1);
   }
}
class Dummy
{
   int data[];
};
void OnStart()
{
   double array2D[][2];
   double fixed[LIMIT];
   double dynamic[];
   MqlRates rates[];
   Dummy dummies[];
   
   ArrayResize(dynamic, LIMIT); // allocating memory
   // fill in a couple of arrays with numbers: 1, 2, 3,...
   indexArray(fixed);
   indexArray(dynamic);
   ...
We have a two-dimensional array array2D, fixed and dynamic array, all of which are of type double, as
well as arrays of structures and class objects. The fixed and dynamic arrays are filled with consecutive
integers (using the auxiliary function indexArray) for demonstration purposes. For other array types of
arrays, we will only check the applicability of the "series" mode, since the idea of the reversal indexing
effect will become clear from the example of filled arrays.
First, make sure none of the arrays are the terminal's own array:
   PRTS(ArrayIsSeries(array2D)); // false
   PRTS(ArrayIsSeries(fixed));   // false
   PRTS(ArrayIsSeries(dynamic)); // false
   PRTS(ArrayIsSeries(rates));   // false
All ArrayIsSeries calls return false since we defined all arrays in the MQL program. We will see the true
value for parameter arrays of the function OnCalculate in indicators (in the fifth Part).
Next, let's check the initial direction of array indexing:

---

## Page 381

Part 4. Common APIs
381 
4.3 Working with arrays
   PRTS(ArrayGetAsSeries(array2D)); // false, cannot be true
   PRTS(ArrayGetAsSeries(fixed));   // false
   PRTS(ArrayGetAsSeries(dynamic)); // false
   PRTS(ArrayGetAsSeries(rates));   // false
   PRTS(ArrayGetAsSeries(dummies)); // false
And again we will get false everywhere.
Let's output arrays fixed and dynamic to the journal to see the original order of the elements.
   ArrayPrint(fixed, 1);
   ArrayPrint(dynamic, 1);
   /*
       1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  9.0 10.0
       1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  9.0 10.0
   */
Now we try to change the indexing order:
   // error: parameter conversion not allowed
   // PRTS(ArraySetAsSeries(array2D, true));
   // warning: cannot be used for static allocated array
   PRTS(ArraySetAsSeries(fixed, true));   // false
   // after this everything is standard
   PRTS(ArraySetAsSeries(dynamic, true)); // true
   PRTS(ArraySetAsSeries(rates, true));   // true
   PRTS(ArraySetAsSeries(dummies, true)); // true
A statement for the array2D array causes a compilation error and is therefore commented out.
A statement for the fixed array issues a compiler warning that it cannot be applied to an array of
constant size. At runtime, all 3 last statements returned success (true). Let's see how the attributes of
the arrays have changed:
   // attribute checks:
   // first, whether they are native to the terminal
   PRTS(ArrayIsSeries(fixed));            // false
   PRTS(ArrayIsSeries(dynamic));          // false
   PRTS(ArrayIsSeries(rates));            // false
   PRTS(ArrayIsSeries(dummies));          // false
   
   // second, indexing direction
   PRTS(ArrayGetAsSeries(fixed));         // false
   PRTS(ArrayGetAsSeries(dynamic));       // true
   PRTS(ArrayGetAsSeries(rates));         // true
   PRTS(ArrayGetAsSeries(dummies));       // true
As expected, the arrays didn't turn into the terminal's own arrays. However, three out of four arrays
changed their indexing to timeseries mode, including an array of structures and objects. To
demonstrate the result, the fixed and dynamic arrays are again displayed in the log.

---

## Page 382

Part 4. Common APIs
382
4.3 Working with arrays
   ArrayPrint(fixed, 1);    // without changes 
   ArrayPrint(dynamic, 1);  // reverse order
   /*
       1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  9.0 10.0
      10.0  9.0  8.0  7.0  6.0  5.0  4.0  3.0  2.0  1.0
   */
Since the mode was not applied to the array of constant size, it remained unchanged. The dynamic
array is now displayed in reverse order.
If you put the array into reverse indexing mode, resize it, and then return the previous indexing, then
the added elements will be inserted at the beginning of the array.
4.3.9 Zeroing objects and arrays
Usually, initialization or filling of variables and arrays does not cause problems. So, for simple variables,
we can simply use the operator '=' in the definition statement along with initialization, or assign the
desired value at any later time.
Aggregate view initialization is available for structures (see section Defining Structures):
Struct struct = {value1, value2, ...};
But it is possible only if there are no dynamic arrays and strings in the structure. Moreover, the
aggregate initialization syntax cannot be used to clean up a structure again. Instead, you must either
assign values to each field individually or reserve an instance of the empty structure in the program and
copy it to clearable instances.
If at the same time, we are talking about an array of structures, then the source code will quickly grow
due to auxiliary but necessary instructions.
For arrays, there are the ArrayInitialize and ArrayFill functions, but they only support numeric types: an
array of strings or structures cannot be filled with them.
In such cases, the ZeroMemory function can be useful. It is not a panacea, since it has significant
limitations in the scope, but it is good to know it.
void ZeroMemory(void &entity)
The function can be applied to a wide range of different entities: variables of simple or object types, as
well as their arrays (fixed, dynamic, or multidimensional).
Variables get the 0 value (for numbers) or its equivalent (NULL for strings and pointers).
In the case of an array, all its elements are set to zero. Do not forget that the elements can be objects,
and in turn, contain objects. In other words, the ZeroMemory function performs a deep memory
cleanup in a single call.
However, there are restrictions on valid objects. You can only populate with zeros the objects of
structures and classes, which:
• contain only public fields (i.e., they do not contain data with access type private or protected)
• do not contain fields with the const modifier
• do not contain pointers

---

## Page 383

Part 4. Common APIs
383
4.3 Working with arrays
The first two restrictions are built into the compiler: an attempt to nullify objects with fields that do not
meet the specified requirements will cause errors (see below).
The third limitation is a recommendation: external zeroing of a pointer will make it difficult to check the
integrity of the data, which is likely to lead to the loss of the associated object and to a memory leak.
Strictly speaking, the requirement of publicity of fields in nullable objects violates the encapsulation
principle, which is inherent in class objects, and therefore ZeroMemory is mainly used with objects of
simple structures and their arrays.
Examples of working with ZeroMemory are given in the script ZeroMemory.mq5.
The problems with the aggregate initialization list are demonstrated using the structure Simple:
#define LIMIT 5
   
struct Simple
{
   MqlDateTime data[]; // dynamic array disables initialization list,
   // string s; // and a string field would also forbid,
   // ClassType *ptr; // and a pointer too
   Simple()
   {
      // allocating memory, it will contain arbitrary data
      ArrayResize(data, LIMIT);
   }
};
In the OnStart function or in the global context, we cannot define and immediately nullify an object of
such a structure:
void OnStart()
{
   Simple simple = {}; // error: cannot be initialized with initializer list
   ...
The compiler throws the error "cannot use initialization list". It is specific to fields like dynamic arrays,
string variables, and pointers. In particular, if the data array were of a fixed size, no error would occur.
Therefore, instead of an initialization list, we use ZeroMemory:
void OnStart()
{
   Simple simple;
   ZeroMemory(simple);
   ...
The initial filling with zeros could also be done in the structure constructor, but it is more convenient to
do subsequent cleanups outside (or provide a method for this with the same function ZeroMemory).
The following class is defined in Base.

---

## Page 384

Part 4. Common APIs
384
4.3 Working with arrays
class Base
{
public: // public is required for ZeroMemory
   // const for any field will cause a compilation error when calling ZeroMemory:
   // "not allowed for objects with protected members or inheritance"
   /* const */ int x;
   Simple t;   // using a nested structure: it will also be nulled
   Base()
   {
      x = rand();
   }
   virtual void print() const
   {
      PrintFormat("%d %d", &this, x);
      ArrayPrint(t.data);
   }
};
Since the class is further used in arrays of objects nullable with ZeroMemory, we are forced to write an
access section public for its fields (which, in principle, is not typical for classes and is done to illustrate
the requirements imposed by ZeroMemory). Also, note that fields cannot have the modifier const.
Otherwise, we'll get a compilation error with text that unfortunately doesn't really fit the problem:
"forbidden for objects with protected members or inheritance".
The class constructor fills the field x with a random number so that later you can clearly see its
cleaning by the function ZeroMemory. The print method displays the contents of all fields for analysis,
including the unique object number (descriptor) &this.
MQL5 does not prevent ZeroMemory from being applied to a pointer variable:
   Base *base = new Base();
   ZeroMemory(base); // will set the pointer to NULL but leave the object
However, this should not be done, because the function resets only the base variable itself, and, if it
referred to an object, this object will remain "hanging" in memory, inaccessible from the program due
to the loss of the pointer.
You can nullify a pointer only after the pointer instance has been freed using the delete operator.
Furthermore, it is easier to reset a separate pointer from the above example, like any other simple
variable (non-composite), using an assignment operator. It makes sense to use ZeroMemory for
composite objects and arrays.
The function allows you to work with objects of the class hierarchy. For example, we can describe the
derivative of the Dummy class derived from Base:

---

## Page 385

Part 4. Common APIs
385
4.3 Working with arrays
class Dummy : public Base
{
public:
   double data[]; // could also be multidimensional: ZeroMemory will work
   string s;
   Base *pointer; // public pointer (dangerous)
   
public:
   Dummy()
   {
      ArrayResize(data, LIMIT);
      
      // due to subsequent application of ZeroMemory to the object
      // we'll lose the 'pointer'
      // and get warnings when the script ends
      // about undeleted objects of type Base
      pointer = new Base();
   }
   
   ~Dummy()
   {
      // due to the use of ZeroMemory, this pointer will be lost
      // and will not be freed
      if(CheckPointer(pointer) != POINTER_INVALID) delete pointer;
   }
   
   virtual void print() const override
   {
      Base::print();
      ArrayPrint(data);
      Print(pointer);
      if(CheckPointer(pointer) != POINTER_INVALID) pointer.print();
   }
};
It includes fields with a dynamic array of type double, string and pointer of type Base (this is the same
type from which the class is derived, but it is used here only to demonstrate the pointer problems, so
as not to describe another dummy class). When the ZeroMemory function nullifies the Dummy object,
an object at pointer is lost and cannot be freed in the destructor. As a result, this leads to warnings
about memory leaks in the remaining objects after the script terminates.
ZeroMemory is used in OnStart to clear the Dummy objects array:

---

## Page 386

Part 4. Common APIs
386
4.3 Working with arrays
void OnStart()
{
   ...
   Print("Initial state");
   Dummy array[];
   ArrayResize(array, LIMIT);
   for(int i = 0; i < LIMIT; ++i)
   {
      array[i].print();
   }
   ZeroMemory(array);
   Print("ZeroMemory done");
   for(int i = 0; i < LIMIT; ++i)
   {
      array[i].print();
   }
The log will output something like the following (the initial state will be different because it prints the
contents of the "dirty", newly allocated memory; here is a small code part):
Initial state
1048576 31539
     [year]     [mon]    [day] [hour] [min] [sec] [day_of_week] [day_of_year]
[0]       0     65665       32      0     0     0             0             0
[1]       0         0        0      0     0     0         65624             8
[2]       0         0        0      0     0     0             0             0
[3]       0         0        0      0     0     0             0             0
[4] 5242880 531430129 51557552      0     0 65665            32             0
0.0 0.0 0.0 0.0 0.0
...
ZeroMemory done
1048576 0
    [year] [mon] [day] [hour] [min] [sec] [day_of_week] [day_of_year]
[0]      0     0     0      0     0     0             0             0
[1]      0     0     0      0     0     0             0             0
[2]      0     0     0      0     0     0             0             0
[3]      0     0     0      0     0     0             0             0
[4]      0     0     0      0     0     0             0             0
0.0 0.0 0.0 0.0 0.0
...
5 undeleted objects left
5 objects of type Base left
3200 bytes of leaked memory
To compare the state of objects before and after cleaning, use descriptors.
So, a single call to ZeroMemory is able to reset the state of an arbitrary branched data structure
(arrays, structures, arrays of structures with nested structure fields and arrays).
Finally, let's see how ZeroMemory can solve the problem of string array initialization. The ArrayInitialize
and ArrayFill functions do not work with strings.

---

## Page 387

Part 4. Common APIs
387
4.3 Working with arrays
   string text[LIMIT] = {};
   // an algorithm populates and uses 'text'
   // ...
   // then you need to re-use the array
   // calling functions gives errors:
   // ArrayInitialize(text, NULL);
   //      `-> no one of the overloads can be applied to the function call
   // ArrayFill(text, 0, 10, NULL);
   //      `->  'string' type cannot be used in ArrayFill function
   ZeroMemory(text);               // ok
In the commented instructions, the compiler would generate errors, stating that the type string is not
supported in these functions.
The way out of this problem is the ZeroMemory function.
4.4 Mathematical functions
The most popular mathematical functions are usually available in all modern programming languages,
and MQL5 is no exception. In this chapter, we'll take a look at several groups of out-of-the-box
functions. These include rounding, trigonometric, hyperbolic, exponential, logarithmic, and power
functions, as well as a few special ones, such as generating random numbers and checking real
numbers for normality.  
Most of the functions have two names: full (with the prefix "Math" and capitalization) and abbreviated
(without a prefix, in lowercase letters). We will provide both options: they work the same way. The
choice can be made based on the formatting style of the source codes.
Since mathematical functions perform some calculations and return a result as a real number, potential
errors can lead to a situation where the result is undefined. For example, you cannot take the square
root of a negative number or take the logarithm of zero. In such cases, the functions return special
values that are not numbers (NaN, Not A Number). We have already faced them in the sections Real
numbers, Arithmetic operations, and Numbers to strings and back. The number correctness and the
absence of errors can be analyzed using the MathIsValidNumber and MathClassify functions (see
section Checking real numbers for normality).
The presence of at least one operand with a value of NaN will cause any subsequent computations
implying this operand, including function calls, to also result in NaN.
For self-study and visual material, you can use the MathPlot.mq5 script as an attachment, which
allows you to display mathematical function graphs with one argument from those described. The
script uses the standard drawing library Graphic.mqh provided in MetaTrader 5 (outside the scope
of this book). Below is a sample of what a hyperbolic sine curve might look like in the MetaTrader 5
window.

---

## Page 388

Part 4. Common APIs
388
4.4 Mathematical functions
Hyperbolic sine chart in the MetaTrader 5 window
4.4.1  The absolute value of a number
The MQL5 API provides the MathAbs function which can remove the minus sign from the number if it
exists. Therefore, there is no need to manually code longer equivalents like this:
if(x < 0) x = -x;
numeric MathAbs(numeric value) ≡ numeric fabs(numeric value)
The function returns the absolute value of the number passed to it, i.e., its modulus. The argument can
be a number of any type. In other words, the function is overloaded for char/uchar, short/ushort,
int/uint, long/ulong, float and double, although for unsigned types the values are always non-negative.
When passing a string, it will be implicitly converted to a double number, and the compiler will generate
a relevant warning.
The type of the return value is always the same as the type of the argument, and therefore the
compiler may need to cast the value to the receiving variable type if the types are different.
Function usage examples are available in the MathAbs.mq5 file.

---

## Page 389

Part 4. Common APIs
389
4.4 Mathematical functions
void OnStart()
{
   double x = 123.45;
   double y = -123.45;
   int i = -1;
   
   PRT(MathAbs(x)); // 123.45, number left "as is"
   PRT(MathAbs(y)); // 123.45, minus sign gone 
   PRT(MathAbs(i)); // 1, int is handled naturally
   
   int k = MathAbs(i);  // no warning: type int for parameter and result
   
   // situations with warnings:
   // double to long conversion required
   long j = MathAbs(x); // possible loss of data due to type conversion
   
   // need to be converted from large type (4 bytes) to small type (2 bytes)
   short c = MathAbs(i); // possible loss of data due to type conversion
   ...
It's important to note that converting a signed integer to an unsigned integer is not equivalent to taking
the modulus of a number:
   uint u_cast = i;
   uint u_abs = MathAbs(i);
   PRT(u_cast);             // 4294967295, 0xFFFFFFFF
   PRT(u_abs);              // 1
Also note that the number 0 can have a sign:
   ...
   double n = 0;
   double z = i * n;
   PRT(z);               // -0.0
   PRT(MathAbs(z));      //  0.0
   PRT(z == MathAbs(z)); // true
}
One of the best examples of how to use MathAbs is to test two real numbers for equality. As is known,
real numbers have a limited accuracy of representing values, which can further degrade in the course
of lengthy calculations (for example, the sum of ten values 0.1  does not give exactly 1 .0). Strict
condition value1  == value2 can give false in most cases, when purely speculative equality should hold.
Therefore, to compare real values, the following notation is usually used:
MathAbs(value1 - value2) < EPS
where EPS is a small positive value which indicates a precision (see an example in the Comparison
operations section).

---

## Page 390

Part 4. Common APIs
390
4.4 Mathematical functions
4.4.2 Maximum and minimum of two numbers
To find the largest or smallest number out of two, MQL5 offers functions MathMax and MathMin. Their
short aliases are respectively fmax and fmin.
n u m er i c M a th M a x ( n u m er i c v a l u e1 , n u m er i c v a l u e2) ≡  n u m er i c fm a x ( n u m er i c v a l u e1 , n u m er i c v a l u e2)
n u m er i c M a th M i n ( n u m er i c v a l u e1 , n u m er i c v a l u e2) ≡  n u m er i c fm i n ( n u m er i c v a l u e1 , n u m er i c v a l u e2)
The functions return the maximum or minimum of the two values passed. The functions are overloaded
for all built-in types.
If parameters of different types are passed to the function, then the parameter of the "lower" type is
automatically converted to the "higher" type, for example, in a pair of types int and double, int will be
brought to double. For more information on implicit type casting, see section Arithmetic type
conversions. The return type corresponds to the "highest" type.
When there is a parameter of type string, it will be "senior", that is, everything is reduced to a string.
Strings will be compared lexicographically, as in the StringCompare function.
The MathMaxMin.mq5 script demonstrates the functions in action.
void OnStart()
{
   int i = 10, j = 11;
   double x = 5.5, y = -5.5;
   string s = "abc";
   
   // numbers   
   PRT(MathMax(i, j)); // 11
   PRT(MathMax(i, x)); // 10
   PRT(MathMax(x, y)); // 5.5
   PRT(MathMax(i, s)); // abc
   
   // type conversions
   PRT(typename(MathMax(i, j))); // int, as is
   PRT(typename(MathMax(i, x))); // double
   PRT(typename(MathMax(i, s))); // string
}
4.4.3 Rounding functions
The MQL5 API includes several functions for rounding numbers to the nearest integer (in one direction
or another). Despite the rounding operation, all functions return a number of type double (with an
empty fractional part).
From a technical point of view, they accept arguments of any numeric type, but only real numbers are
rounded, and integers are only converted to double.
If you want to round up to a specific sign, use NormalizeDouble (see section Normalization of doubles).
Examples of working with functions are given in the file MathRound.mq5.

---

## Page 391

Part 4. Common APIs
391 
4.4 Mathematical functions
double MathRound(numeric value) ≡ double round(numeric value)
The function rounds a number up or down to the nearest integer.
   PRT((MathRound(5.5)));  // 6.0
   PRT((MathRound(-5.5))); // -6.0
   PRT((MathRound(11)));   // 11.0
   PRT((MathRound(-11)));  // -11.0
If the value of the fractional part is greater than or equal to 0.5, the mantissa is increased by one
(regardless of the sign of the number).
double MathCeil(numeric value) ≡ double ceil(numeric value)
double MathFloor(numeric value) ≡ double floor(numeric value)
The functions return the closest greater integer value (for ceil) or closest lower integer value (for floor)
to the transferred value. If value is already equal to an integer (has a zero fractional part), this integer
is returned.
   PRT((MathCeil(5.5)));   // 6.0
   PRT((MathCeil(-5.5)));  // -5.0
   PRT((MathFloor(5.5)));  // 5.0
   PRT((MathFloor(-5.5))); // -6.0
   PRT((MathCeil(11)));    // 11.0
   PRT((MathCeil(-11)));   // -11.0
   PRT((MathFloor(11)));   // 11.0
   PRT((MathFloor(-11)));  // -11.0
4.4.4 Remainder after division (Modulo operation)
To divide integers with remainder, MQL5 has the built-in modulo operator '%', described in the section
Arithmetic operations. However, this operator is not applicable to real numbers. In the case when the
divisor, the dividend, or both operands are real, you should use the function MathMod (or short form
fmod).
d o u b l e M a th M o d ( d o u b l e d i v i d en d , d o u b l e d i v i d er ) ≡  d o u b l e fm o d ( d o u b l e d i v i d en d , d o u b l e d i v i d er )
The function returns the real remainder after dividing the first passed number (dividend) by the second
(divider).
If any argument is negative, the sign of the result is determined by the rules described in the above
section.
Examples of how the function works are available in the script MathMod.mq5.
   PRT(MathMod(10.0, 3));     // 1.0
   PRT(MathMod(10.0, 3.5));   // 3.0
   PRT(MathMod(10.0, 3.49));  // 3.02
   PRT(MathMod(10.0, M_PI));  // 0.5752220392306207
   PRT(MathMod(10.0, -1.5));  // 1.0, the sign is gone
   PRT(MathMod(-10.0, -1.5)); // -1.0

---

## Page 392

Part 4. Common APIs
392
4.4 Mathematical functions
4.4.5 Powers and roots
The MQL5 API provides a generic function MathPow for raising a number to an arbitrary power, as well
as a function for a special case with a power of 0.5, more familiar to us as the operation of extracting a
square root MathSqrt.
To test the functions, use the MathPowSqrt.mq5 script.
double MathPow(double base, double exponent) ≡ double pow(double base, double exponent)
The function raises the base to the specified power exponent.
   PRT(MathPow(2.0, 1.5));  // 2.82842712474619
   PRT(MathPow(2.0, -1.5)); // 0.3535533905932738
   PRT(MathPow(2.0, 0.5));  // 1.414213562373095
double MathSqrt(double value) ≡ double sqrt(double value)
The function returns the square root of a number.
   PRT(MathSqrt(2.0));      // 1.414213562373095
   PRT(MathSqrt(-2.0));     // -nan(ind)
MQL5 defines several constants containing ready-made calculation values involving sqrt.
Constant
Description
Value
M_SQRT2
sqrt(2.0)
1 .41 421 356237309504
880
M_SQRT1 _2
1  / sqrt(2.0)
0.7071 06781 1 8654752
4401 
M_2_SQRTPI
2.0 / sqrt(M_PI)
1 .1 283791 6709551 257
390
Here M_PI is the Pi number (π=3.1 41 59265358979323846, see further along the section
Trigonometric functions).
All built-in constants are described in the documentation.
4.4.6 Exponential and logarithmic functions
Calculation of exponential and logarithmic functions is available in MQL5 using the corresponding API
section.
The absence of the binary logarithm in the API, which is often required in computer science and
combinatorics, is not a problem, since it is easy to calculate, upon request, through the available
natural or decimal logarithm functions.

---

## Page 393

Part 4. Common APIs
393
4.4 Mathematical functions
log2(x) = log(x) / log(2) = log(x) / M_LN2
log2(x) = log10(x) / log10(2)
Here log and log1 0 are available logarithmic functions (based on e and 1 0, respectively), M_LN2 is a
built-in constant equal to log(2).
The following table lists all the constants that can be useful in logarithmic calculations.
Constant
Description
Value
M_E
e
2.71 8281 82845904523536
M_LOG2E
log2(e)
1 .44269504088896340736
M_LOG1 0E
log1 0(e)
0.434294481 903251 82765
1
M_LN2
ln(2)
0.6931 471 8055994530941 
7
M_LN1 0
ln(1 0)
2.30258509299404568402
Examples of the functions described below are collected in the file MathExp.mq5.
double MathExp(double value) ≡ double exp(double value)
The function returns the exponent, i.e., the number e (available as a predefined constant M_E) raised to
the specified power value. On overflow, the function returns inf (a kind of NaN for infinity).
   PRT(MathExp(0.5));      // 1.648721270700128
   PRT(MathPow(M_E, 0.5)); // 1.648721270700128
   PRT(MathExp(10000.0));  // inf, NaN
 
double MathLog(double value) ≡ double log(double value)
The function returns the natural logarithm of the passed number. If value is negative, the function
returns -nan(ind) (NaN "undefined value"). If value is 0, the function returns inf (NaN "infinity").
   PRT(MathLog(M_E));     // 1.0
   PRT(MathLog(10000.0)); // 9.210340371976184
   PRT(MathLog(0.5));     // -0.6931471805599453
   PRT(MathLog(0.0));     // -inf, NaN
   PRT(MathLog(-0.5));    // -nan(ind)
   PRT(Log2(128));        // 7
The last line uses the implementation of the binary logarithm through MathLog:
double Log2(double value)
{
   return MathLog(value) / M_LN2;
}

---

## Page 394

Part 4. Common APIs
394
4.4 Mathematical functions
double MathLog1 0(double value) ≡ double log1 0(double value)
The function returns the decimal logarithm of a number.
   PRT(MathLog10(10.0)); // 1.0
   PRT(MathLog10(10000.0)); // 4.0
double MathExpm1 (double value) ≡ double expm1 (double value)
The function returns the value of the expression (MathExp(value) - 1 ). In economic calculations, the
function is used to calculate the effective interest (revenue or payment) per unit of time in a compound
interest scheme when the number of periods tends to infinity.
   PRT(MathExpm1(0.1)); // 0.1051709180756476
double MathLog1 p(double value) ≡ double log1 p(double value)
The function returns the value of the expression MathLog(1  + value), i.e., it performs the opposite
action to the function MathExpm1 .
   PRT(MathLog1p(0.1)); // 0.09531017980432487
4.4.7 Trigonometric functions
MQL5 provides the three main trigonometric functions (MathCos, MathSin, MathTan) and their inverses
(MathArccos, MathArcsin, MathArctan). They all work with angles in radians. For angles in degrees, use
the formula:
radians = degrees * M_PI / 180
Here M_PI is one of several constants with trigonometric quantities (pi and its derivatives) built into the
language.
Constant
Description
Value
M_PI
π
3.1 41 59265358979323846
M_PI_2
π/2
1 .57079632679489661 923
M_PI_4
π/4
0.7853981 6339744830961 
6
M_1 _PI
1 /π
0.31 83098861 83790671 53
8
M_2_PI
2/π
0.63661 9772367581 34307
6
The arc tangent can also be calculated for a quantity represented by the ratio of two coordinates y and
x: this extended version is called MathArctan2; it is able to restore angles in the full range of the circle
from -M_PI to +M_PI, unlike MathArctan, which is limited to -M_PI_2 to +M_PI_2.

---

## Page 395

Part 4. Common APIs
395
4.4 Mathematical functions
Trigonometric functions and quadrants of the unit circle
Examples of calculations are given in the script MathTrig.mq5 (see after the descriptions).
double MathCos(double value) ≡ double cos(double value)
double MathSin(double value) ≡ double sin(double value)
The functions return, respectively, the cosine and sine of the passed number (the angle is in radians).
double MathTan(double value) ≡ double tan(double value)
The function returns the tangent of the passed number (the angle is in radians).
double MathArccos(double value) ≡ double acos(double value)
double MathArcsin(double value) ≡ double asin(double value)
The functions return the value, respectively, of the arc cosine and arc sine of the passed number, i.e.,
the angle in radians. If x = MathCos(t), then t = MathArccos(x). The sine and arcsine have a similar
scheme. If y = MathSin(t), then t = MathArcsin(y).
The parameter must be between -1  and +1 . Otherwise, the function will return NaN.
The result of the arccosine is in the range from 0 to M_PI, and the result of the arcsine is from -
M_PI_2 to +M_PI_2. The indicated ranges are called the main ranges, since the functions are multi-
valued, i.e., their values are periodically repeated. The selected half-periods completely cover the
definition area from -1  to +1 .

---

## Page 396

Part 4. Common APIs
396
4.4 Mathematical functions
The resulting angle for the cosine lies in the upper semicircle, and the symmetric solution in the lower
semicircle can be obtained by adding a sign, i.e.t=-t. For the sine, the resulting angle is in the right
semicircle, and the second solution in the left semicircle is M_ PI-t (if for negative t it is also required to
obtain a negative additional angle, then -M_ PI-t).
double MathArctan(double value) ≡ double atan(double value)
The function returns the value of the arc tangent for the passed number, i.e., the angle in radians, in
the range from -M_PI_2 to +M_PI_2.
The function is inverse to MathTan, but with one caveat.
Please note that the period of the tangent is 2 times less than the full period (circumference) due to
the fact that the ratio of sine and cosine is repeated in opposite quadrants (quarters of a circle) due to
superposition of signs. As a result, the tangent value alone is not sufficient to uniquely determine the
original angle over the full range from -M_PI to +M_PI. This can be done using the function
MathArctan2, in which the tangent is represented by two separate components.
 
double MathArctan2(double y, double x) ≡ double atan2(double y, double x)
The function returns in radians the value of the angle, the tangent of which is equal to the ratio of two
specified numbers: coordinates along the y axis and along the x axis.
The result (let's denote it as r) lies in the range from -M_PI to +M_PI, and the condition MathTan(r) = y
/ x is met for it.
The function takes into account the sign of both arguments to determine the correct quadrant (subject
to boundary conditions, when either x, or y are equal to 0, that is, they are on the border of the
quadrants).
• 1  — x >= 0, y >= 0, 0 <= r <= M_PI_2
• 2 — x < 0, y >= 0, M_PI_2 < r <= M_PI
• 3 — x < 0, y < 0, -M_PI < r < -M_PI_2
• 4 — x >= 0, y < 0, -M_PI_2 <= r < 0
Below are the results of calling trigonometric functions in the script MathTrig.mq5.
void OnStart()
{
   PRT(MathCos(1.0));     // 0.5403023058681397
   PRT(MathSin(1.0));     // 0.8414709848078965
   PRT(MathTan(1.0));     // 1.557407724654902
   PRT(MathTan(45 * M_PI / 180.0)); // 0.9999999999999999
   
   PRT(MathArccos(1.0));         // 0.0
   PRT(MathArcsin(1.0));         // 1.570796326794897 == M_PI_2
   PRT(MathArctan(0.5));         // 0.4636476090008061, Q1
   PRT(MathArctan2(1.0, 2.0));   // 0.4636476090008061, Q1
   PRT(MathArctan2(-1.0, -2.0)); // -2.677945044588987, Q3
}

---

## Page 397

Part 4. Common APIs
397
4.4 Mathematical functions
4.4.8 Hyperbolic functions
The MQL5 API includes a set of direct and inverse hyperbolic functions.
Hyperbolic functions
double MathCosh(double value) ≡ double cosh(double value)
double MathSinh(double value) ≡ double sinh(double value)
double MathTanh(double value) ≡ double tanh(double value)
The three basic functions calculate the hyperbolic cosine, sine and tangent.
 
double MathArccosh(double value) ≡ double acosh(double value)
double MathArcsinh(double value) ≡ double asinh(double value)
double MathArctanh(double value) ≡ double atanh(double value)
The three inverse functions calculate the hyperbolic inverse cosine, inverse sine, and arc tangent.
For the arc cosine, the argument must be greater than or equal to +1 . Otherwise, the function will
return NaN.
The arc tangent is defined from -1  to +1 . If the argument is beyond these limits, the function will
return NaN.
Examples of hyperbolic functions are shown in the MathHyper.mq5 script.

---

## Page 398

Part 4. Common APIs
398
4.4 Mathematical functions
void OnStart()
{
   PRT(MathCosh(1.0));    // 1.543080634815244
   PRT(MathSinh(1.0));    // 1.175201193643801
   PRT(MathTanh(1.0));    // 0.7615941559557649
   
   PRT(MathArccosh(0.5)); // nan
   PRT(MathArcsinh(0.5)); // 0.4812118250596035
   PRT(MathArctanh(0.5)); // 0.5493061443340549
   
   PRT(MathArccosh(1.5)); // 0.9624236501192069
   PRT(MathArcsinh(1.5)); // 1.194763217287109
   PRT(MathArctanh(1.5)); // nan
}
4.4.9 Normality test for real numbers
Since calculations with real numbers can have abnormal situations, such as going beyond the scope of
a function, obtaining mathematical infinity, lost order, and others, the result may not contain a number.
Instead, it may contain a special value that actually describes the nature of the problem. All such
special values have a generic name "not a number" (Not A Number, NaN).
We have already faced them in the previous sections of the book. In particular, when outputting to a
journal (see section Numbers to strings and vice versa) they are displayed as text labels (for example,
nan(ind), +inf and others). Another feature is that a single NaN value among the operands of any
expression is enough for the entire expression to stop evaluating correctly and begin to give the result
NaN. The only exceptions are "non-numbers" representing the plus/minus of infinity: if you divide
something by them, you get zero. However, there is an expected exception here: if we divide infinity by
infinity, we again get NaN.
Therefore, it is important for programs to determine the moment when NaN appears in the calculations
and handle the situation in a special way: signal an error, substitute some acceptable default value, or
repeat the calculation with other parameters (for example, reduce the accuracy/step of the iterative
algorithm).
There are 2 functions in MQL5 that allow you to analyze a real number for
normality:MathIsValidNumber gives a simple answer: yes (true) or not (false), and MathClassify
produces more detailed categorization.
At the physical level, all special values are encoded in a number with a special combination of bits that
is not used to represent ordinary numbers. For types double and float these encodings are, of course,
different. Let's take a look at double behind the scenes (as it is more in demand than float).
In the chapter Nested templates, we created a Converter class for switching views by combining two
different types in a union. Let's use this class to study the NaN bit device.
For convenience, we will move the class to a separate header file ConverterT.mqh. Let's connect this
mqh-file in the test script MathInvalid.mq5 and create an instance of a converter for a bunch of types
double/ulong (the order is not important as the converter is able to work in both directions).

---

## Page 399

Part 4. Common APIs
399
4.4 Mathematical functions
static Converter<ulong, double> NaNs;
The combination of bits in NaN is standardized, so let's take a few commonly used values represented
by constants ulong, and see how the built-in functions react to them.
// basic NaNs
#define NAN_INF_PLUS  0x7FF0000000000000
#define NAN_INF_MINUS 0xFFF0000000000000
#define NAN_QUIET     0x7FF8000000000000
#define NAN_IND_MINUS 0xFFF8000000000000
   
// custom NaN examples
#define NAN_QUIET_1   0x7FF8000000000001
#define NAN_QUIET_2   0x7FF8000000000002
   
static double pinf = NaNs[NAN_INF_PLUS];  // +infinity
static double ninf = NaNs[NAN_INF_MINUS]; // -infinity
static double qnan = NaNs[NAN_QUIET];     // quiet NaN
static double nind = NaNs[NAN_IND_MINUS]; // -nan(ind)
   
void OnStart()
{
   PRT(MathIsValidNumber(pinf));               // false
   PRT(EnumToString(MathClassify(pinf)));      // FP_INFINITE
   PRT(MathIsValidNumber(nind));               // false
   PRT(EnumToString(MathClassify(nind)));      // FP_NAN
   ...
}
As expected, the results were the same.
Let's view the formal description of the MathIsValidNumber and MathClassify functions and then
continue with the tests.
bool MathIsValidNumber(double value)
The function checks the correctness of a real number. The parameter can be of type double or float.
The resulting true means the correct number, and false means "not a number" (one of the varieties of
NaN).
ENUM_FP_CLASS MathClassify(double value)
The function returns the category of a real number (of type double or float) which is one of the enum
ENUM_FP_CLASS values:
• FP_NORMAL is a normal number.
• FP_SUBNORMAL is a number less than the minimum number representable in a normalized form
(for example, for the type double these are values less than DBL_MIN, 2.225073858507201 4e-
308); loss of order (accuracy).
• FP_ZERO is zero (positive or negative).
• FP_INFINITE is infinity (positive or negative).
• FP_NAN means all other types of "non-numbers" (subdivided into families of "silent" and "signal"
NaN).

---

## Page 400

Part 4. Common APIs
400
4.4 Mathematical functions
MQL5 does not provide alerting NaNs which are used in the exceptions mechanism and allows the
interception and response to critical errors within the program. There is no such mechanism in MQL5,
so, for example, in case of a zero division, the MQL program simply terminates its work (unloads from
the chart).
There can be many "quiet" NaNs, and you can construct them using a converter to differentiate and
handle non-standard states in your computational algorithms.
Let's perform some calculations in MathInvalid.mq5 to visualize how the numbers of different categories
can be obtained.
 // calculations with double
   PRT(MathIsValidNumber(0));                      // true
   PRT(EnumToString(MathClassify(0)));             // FP_ZERO
   PRT(MathIsValidNumber(M_PI));                   // true
   PRT(EnumToString(MathClassify(M_PI)));          // FP_NORMAL
   PRT(DBL_MIN / 10);                              // 2.225073858507203e-309
   PRT(MathIsValidNumber(DBL_MIN / 10));           // true
   PRT(EnumToString(MathClassify(DBL_MIN / 10)));  // FP_SUBNORMAL
   PRT(MathSqrt(-1.0));                            // -nan(ind)
   PRT(MathIsValidNumber(MathSqrt(-1.0)));         // false
   PRT(EnumToString(MathClassify(MathSqrt(-1.0))));// FP_NAN
   PRT(MathLog(0));                                // -inf
   PRT(MathIsValidNumber(MathLog(0)));             // false
   PRT(EnumToString(MathClassify(MathLog(0))));    // FP_INFINITE
   
 // calculations with float
   PRT(1.0f / FLT_MIN / FLT_MIN);                             // inf
   PRT(MathIsValidNumber(1.0f / FLT_MIN / FLT_MIN));          // false
   PRT(EnumToString(MathClassify(1.0f / FLT_MIN / FLT_MIN))); // FP_INFINITE
We can use the converter in the opposite direction: to get its bit representation by value double, and
thereby detect "non-numbers":
   PrintFormat("%I64X", NaNs[MathSqrt(-1.0)]);      // FFF8000000000000
   PRT(NaNs[MathSqrt(-1.0)] == NAN_IND_MINUS);      // true, nind
The PrintFormat function is similar to StringFormat; the only difference is that the result is immediately
printed to the log, and not to a string.
Finally, let's make sure that "not numbers" are always not equal:
   // NaN != NaN always true
   PRT(MathSqrt(-1.0) != MathSqrt(-1.0)); // true
   PRT(MathSqrt(-1.0) == MathSqrt(-1.0)); // false
To get NaN or infinity in MQL5, there is a method based on casting the strings "nan" and "inf" to
double.
double nan = (double)"nan";
double infinity = (double)"inf";

---

