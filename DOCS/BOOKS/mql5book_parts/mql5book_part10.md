# MQL5 Book - Part 10 (Pages 1801-2000)

## Page 1801

Part 7. Advanced language tools
1 801 
7.5 Network functions
void OnStart()
{
   uchar data[], result[];
   string response;
   
   int code = PRTF(WebRequest(Method, Address, Headers, Timeout, data, result, response));
   if(code > -1)
   {
      Print(response);
      if(ArraySize(result) > 0)
      {
         PrintFormat("Got data: %d bytes", ArraySize(result));
         if(DumpDataToFiles)
         {
            string parts[];
            URL::parse(Address, parts);
            
            const string filename = parts[URL_HOST] +
               (StringLen(parts[URL_PATH]) > 1 ? parts[URL_PATH] : "/_index_.htm");
            Print("Saving ", filename);
            PRTF(FileSave(filename, result));
         }
         else
         {
            Print(CharArrayToString(result, 0, 80, CP_UTF8));
         }
      }
   }
}
To form the file name, we use the URL helper class from the header file URL.mqh (which will not be fully
described here). Method URL::parse parses the passed string into URL components according to the
specification as the general form of the URL is always "protocol://domain.com:port/path?query#hash";
note that many fragments are optional. The results are placed in the receiving array, the indexes in
which correspond to specific parts of the URL and are described in the URL_PARTS enumeration:
enum URL_PARTS
{
   URL_COMPLETE,   // full address
   URL_SCHEME,     // protocol
   URL_USER,       // username/password (deprecated, not supported)
   URL_HOST,       // server
   URL_PORT,       // port number
   URL_PATH,       // path/directories
   URL_QUERY,      // query string after '?'
   URL_FRAGMENT,   // fragment after '#' (not highlighted)
   URL_ENUM_LENGTH
};
Thus, when the received data should be written to a file, the script creates it in a folder named after
the server (parts[URL_ HOST]) and so on, preserving the path hierarchy in the URL (parts[URL_ PATH]):

---

## Page 1802

Part 7. Advanced language tools
1 802
7.5 Network functions
in the simplest case, this will simply be the name of the "endpoint". When the home page of a site is
requested (the path contains only a slash '/'), the file is named "_index_.htm".
Let's try to run the script with default parameters, remembering to allow this server in the terminal
settings first. In the log, we will see the following lines (HTTP headers of the server response and a
message about the successful saving of the file):
WebRequest(Method,Address,Headers,Timeout,data,result,response)=200 / ok
Date: Fri, 22 Jul 2022 08:45:03 GMT
Content-Type: application/json
Content-Length: 291
Connection: keep-alive
Server: gunicorn/19.9.0
Access-Control-Allow-Origin: *
Access-Control-Allow-Credentials: true
   
Got data: 291 bytes
Saving httpbin.org/headers
FileSave(filename,result)=true / ok
The httpbin.org/headers file contains the headers of our request as seen by the server (the server
added the JSON formatting itself when answering us).
{
  "headers":
  {
    "Accept": "*/*", 
    "Accept-Encoding": "gzip, deflate", 
    "Accept-Language": "ru,en", 
    "Host": "httpbin.org", 
    "User-Agent": "MetaTrader 5 Terminal/5.3333 (Windows NT 10.0; Win64; x64)", 
    "X-Amzn-Trace-Id": "Root=1-62da638f-2554..." // <- this is added by the reverse proxy server
  }
}
Thus, the terminal reports that it is ready to accept data of any type, with support for compression by
specific methods and a list of preferred languages. In addition, it appears in the User-Agent field as
MetaTrader 5. The latter may be undesirable when working with some sites that are optimized to work
exclusively with browsers. Then we can specify a fictitious name in the headers input parameter, for
example, "User-Agent: Mozilla/5.0 (Windows NT 1 0.0) AppleWebKit/537.36 (KHTML, like Gecko)
Chrome/1 03.0.0.0 Safari/537.36".
Some of the test sites listed above allow you to organize a temporary test environment on the
server with a random name for your personal experiment: to do this, you need to go to the site from
a browser and get a unique link that usually works for 24 hours. Then you will be able to use this
link as an address for requests from MQL5 and monitor the behavior of requests directly from the
browser. There you can also configure server responses, in particular, attempt submitting forms.
Let's make this example slightly more difficult. The server may require additional actions from the
client to fulfill the request, in particular, authorize, perform a "redirect" (go to a different address),
reduce the frequency of requests, etc. All such "signals" are denoted by special HTTP codes returned
by the WebRequest function. For example, codes 301  and 302 mean redirect for different reasons, and
WebRequest executes it internally automatically, re-requesting the page at the address specified by the
server (therefore, redirect codes never end up in the MQL program code). The 401  code requires the

---

## Page 1803

Part 7. Advanced language tools
1 803
7.5 Network functions
client to provide a username and password, and here the entire responsibility lies with us. There are
many ways to send this data. A new script WebRequestAuth.mq5 demonstrates the handling of two
authorization options that the server requests using HTTP response headers: "WWW-Authenticate:
Basic" or "WWW-Authenticate: Digest". In headers it might look like this:
WWW-Authenticate:Basic realm="DemoBasicAuth"
Or like this:
WWW-Authenticate:Digest realm="DemoDigestAuth",qop="auth", »
»  nonce="cuFAuHbb5UDvtFGkZEb2mNxjqEG/DjDr",opaque="fyNjGC4x8Zgt830PpzbXRvoqExsZeQSDZj"
The first of them is the simplest and most unsafe, and therefore is practically not used: it is given in the
book because of how easy it is to learn it at the first stage. The bottom line of its work is to generate
the following HTTP request in response to a server request by adding a special header:
Authorization: Basic dXNlcjpwYXNzd29yZA==
Here, the "Basic" keyword is followed by the Base64-encoded string "user:password" with the actual
username and password, and the ':' character is inserted hereinafter "as is" as a linking block. More
clearly, the interaction process is shown in the image.
Simple authorization scheme on a web server
The authorization scheme Digest is considered more advanced. In this case, the server provides some
additional information in its response:
• realms – the name of the site (site area) where the entry is made
• qop – a variation of the Digest method (we will only consider "auth")
• nonce – a random string that will be used to generate authorization data
• opaque – a random string that we will pass back "as is" in our headers

---

## Page 1804

Part 7. Advanced language tools
1 804
7.5 Network functions
• algorithm – an optional name of the hashing algorithm, MD5 is assumed by default
For authorization, you need to perform the following steps:
1 .Generate your own random string cnonce
2. Initialize or increment your request counter nc
3. Calculate hash1  = MD5(user:realm:password)
4. Calculate hash2 = MD5(method:uri), here uri is the path and name of the page
5. Calculate response = MD5(hash1 :nonce:nc:cnonce:qop:hash2)
After that, the client can repeat the request to the server, adding a line like this to its headers:
Authorization: Digest username="user",realm="realm",nonce="...", »
»  uri="/path/to/page",qop=auth,nc=00000001,cnonce="...",response="...",opaque="..."
Since the server has the same information as the client, it will be able to repeat the calculations and
check the hashes match.
Let's add variables to the script parameters to enter the username and password. By default, the
Address parameter includes the address of the digest-auth endpoint, which can request authorization
with parameters qop ("auth"), login ("test"), and password ("pass"). This is all optional in the endpoint
path (you can test other methods and user credentials, like so: "https://httpbin.org/digest-auth/auth-
int/mql5client/mql5password").
const string Method = "GET";
input string Address = "https://httpbin.org/digest-auth/auth/test/pass";
input string Headers = "User-Agent: noname";
input int Timeout = 5000;
input string User = "test";
input string Password = "pass";
input bool DumpDataToFiles = true;
We specified a dummy browser name in the Headers parameter to demonstrate the feature.
In the OnStart function, we add the processing of HTTP code 401 . If a username and password are not
provided, we will not be able to continue.

---

## Page 1805

Part 7. Advanced language tools
1 805
7.5 Network functions
void OnStart()
{
   string parts[];
   URL::parse(Address, parts);
   uchar data[], result[];
   string response;
   int code = PRTF(WebRequest(Method, Address, Headers, Timeout, data, result, response));
   Print(response);
   if(code == 401)
   {
      if(StringLen(User) == 0 || StringLen(Password) == 0)
      {
         Print("Credentials required");
         return;
      }
      ...
The next step is to analyze the headers received from the server. For convenience, we have written the
HttpHeader class (HttpHeader.mqh). The full text is passed to its constructor, as well as the element
separator (in this case, the newline character '\n') and the character used between the name and
value within each element (in this case, the colon ':'). During its creation, the object "parses" the text,
and then the elements are made available through the overloaded operator [], with the type of its
argument being a string. As a result, we can check for an authorization requirement by the name
"WWW-Authenticate". If such an element exists in the text and is equal to "Basic", we form the
response header "Authorization: Basic" with the login and password encoded in Base64.
      code = -1;
      HttpHeader header(response, '\n', ':');
      const string auth = header["WWW-Authenticate"];
      if(StringFind(auth, "Basic ") == 0)
      {
         string Header = Headers;
         if(StringLen(Header) > 0) Header += "\r\n";
         Header += "Authorization: Basic ";
         Header += HttpHeader::hash(User + ":" + Password, CRYPT_BASE64);
         PRTF(Header);
         code = PRTF(WebRequest(Method, Address, Header, Timeout, data, result, response));
         Print(response);
      }
      ...
For Digest authorization, everything is a little more complicated, following the algorithm outlined above.

---

## Page 1806

Part 7. Advanced language tools
1 806
7.5 Network functions
      else if(StringFind(auth, "Digest ") == 0)
      {
         HttpHeader params(StringSubstr(auth, 7), ',', '=');
         string realm = HttpHeader::unquote(params["realm"]);
         if(realm != NULL)
         {
            string qop = HttpHeader::unquote(params["qop"]);
            if(qop == "auth")
            {
               string h1 = HttpHeader::hash(User + ":" + realm + ":" + Password);
               string h2 = HttpHeader::hash(Method + ":" + parts[URL_PATH]);
               string nonce = HttpHeader::unquote(params["nonce"]);
               string counter = StringFormat("%08x", 1);
               string cnonce = StringFormat("%08x", MathRand());
               string h3 = HttpHeader::hash(h1 + ":" + nonce + ":" + counter + ":" +
                  cnonce + ":" + qop + ":" + h2);
               
               string Header = Headers;
               if(StringLen(Header) > 0) Header += "\r\n";
               Header += "Authorization: Digest ";
               Header += "username=\"" + User + "\",";
               Header += "realm=\"" + realm + "\",";
               Header += "nonce=\"" + nonce + "\",";
               Header += "uri=\"" + parts[URL_PATH] + "\",";
               Header += "qop=" + qop + ",";
               Header += "nc=" + counter + ",";
               Header += "cnonce=\"" + cnonce + "\",";
               Header += "response=\"" + h3 + "\",";
               Header += "opaque=" + params["opaque"] + "";
               PRTF(Header);
               code = PRTF(WebRequest(Method, Address, Header, Timeout, data, result, response));
               Print(response);
            }
         }
      }
Static method HttpHeader::hash gets a string with a hexadecimal hash representation (default MD5) for
all required compound strings. Based on this data, the header is formed for the next WebRequest call.
The static HttpHeader::unquote method removes the enclosing quotes.
The rest of the script remained unchanged. A repeated HTTP request may succeed, and then we will
get the content of the secure page, or authorization will be denied, and the server will write something
like "Access denied".
Since the default parameters contain the correct values ("/digest-auth/auth/test/pass" corresponds to
the user "test" and the password "pass"), we should get the following result of running the script (all
main steps and data are logged).

---

## Page 1807

Part 7. Advanced language tools
1 807
7.5 Network functions
WebRequest(Method,Address,Headers,Timeout,data,result,response)=401 / ok
Date: Fri, 22 Jul 2022 10:45:56 GMT
Content-Type: text/html; charset=utf-8
Content-Length: 0
Connection: keep-alive
Server: gunicorn/19.9.0
WWW-Authenticate: Digest realm="me@kennethreitz.com" »
»  nonce="87d28b529a7a8797f6c3b81845400370", qop="auth",
»  opaque="4cb97ad7ea915a6d24cf1ccbf6feeaba", algorithm=MD5, stale=FALSE
...
The first WebRequest call has ended with code 401 , and among the response headers is an
authorization request ("WWW-Authenticate") with the required parameters. Based on them, we
calculated the correct answer and prepared headers for a new request.
Header=User-Agent: noname
Authorization: Digest username="test",realm="me@kennethreitz.com" »
»  nonce="87d28b529a7a8797f6c3b81845400370",uri="/digest-auth/auth/test/pass",
»  qop=auth,nc=00000001,cnonce="00001c74",
»  response="c09e52bca9cc90caf9a707d046b567b2",opaque="4cb97ad7ea915a6d24cf1ccbf6feeaba" / ok
...
The second request returns 200 and a payload that we write to the file.
WebRequest(Method,Address,Header,Timeout,data,result,response)=200 / ok
Date: Fri, 22 Jul 2022 10:45:56 GMT
Content-Type: application/json
Content-Length: 47
Connection: keep-alive
Server: gunicorn/19.9.0
...
Got data: 47 bytes
Saving httpbin.org/digest-auth/auth/test/pass
FileSave(filename,result)=true / ok
Inside the file MQL5/Files/httpbin.org/digest-auth/auth/test/pass you can find the "web page", or
rather the status of successful authorization in JSON format.
{
  "authenticated": true, 
  "user": "test"
}
If you specify an incorrect password when running the script, we will receive an empty response from
the server, and the file will not be written.
Using WebRequest, we automatically enter the field of distributed software systems, in which the
correct operation depends not only on our client MQL code but also on the server (not to mention
intermediate links, like a proxy). Therefore, you need to be prepared for the occurrence of other
people's mistakes. In particular, at the time of writing the book in the implementation of the digest-
auth endpoint on httpbin.org there was a problem: the username entered in the request did not
participate in the authorization check, and therefore any login leads to successful authorization if
the correct password is specified. Still, to check our script, use other services, for example,

---

## Page 1808

Part 7. Advanced language tools
1 808
7.5 Network functions
something like httpbingo.org/digest-auth/auth/test/pass. You can also configure the script to the
address j igsaw.w3.org/HTTP/Digest/ – it expects login/password "guest"/"guest".
In practice, most sites implement authorization using forms embedded directly in web pages: inside the
HTML code, they are essentially the form container tag with a set of input fields, which are filled in by
the user and sent to the server using the POST method. In this regard, it makes sense to analyze the
example of submitting a form. However, before getting into this in detail, it is desirable to highlight one
more technique.
The thing is that the interaction between the client and the server is usually accompanied by a change
in the state of both the client and the server. Using the example of authorization, this can be
understood most clearly, since before authorization the user was unknown to the system, and after
that, the system already knows the login and can apply the preferred settings for the site (for example,
language, color, forum display method), and also allow access to those pages where unauthorized
visitors cannot get into (the server stops such attempts by returning HTTP status 403, Forbidden).
Support and synchronization of the consistent state of the client and server parts of a distributed web
application is provided using the cookies mechanism which implies named variables and their values in
HTTP headers. The term goes back to "fortune cookies" because cookies also contain small messages
invisible to the user.
Either side, server and client, can add cookie to the HTTP header. The server does this with a line like:
Set-Cookie: name=value; ⌠Domain=domain; Path=path; Expires=date; Max-Age=number_of_seconds ...⌡ᵒᵖᵗ
Only the name and value are required and the rest of the attributes are optional: here are the main
ones − Domain, Path, Expires, and Max age, but in real situations, there are more of them.
Having received such a header (or several headers), the client must remember the name and value of
the variable and send them to the server in all requests that address to the corresponding Domain and
Path inside this domain until the expiration date (Expires or Max-Age).
In an outgoing HTTP request from a client, cookies are passed as a string:
Cookie: name⁽№⁾=value⁽№⁾ ⌠; name⁽ⁱ⁾=value⁽ⁱ⁾ ...⌡ᵒᵖᵗ
Here, separated by a semicolon and a space, all name=value pairs are listed; they are set by the server
and known to this client, matched with the current request by the domain and path, and not expired.
The server and client exchange all the necessary cookies with each HTTP request, which is why this
architectural style of distributed systems is called REST (Representational State Transfer). For
example, after a user successfully logs in to the server, the latter sets (via the "Set-Cookie:" header) a
special "cookie" with the user's identifier, after which the web browser (or, in our case, a terminal with
an MQL program) will send it in subsequent requests (by adding the appropriate line to the "Cookie:"
header).
The WebRequest function silently does all this work for us: collects cookies from incoming headers and
adds appropriate cookies to outgoing HTTP requests.
Cookies are stored by the terminal and between sessions, according to their settings. To check this, it
is enough to request a web page twice from a site using cookies.
Attention, cookies are stored in relation to the site and therefore are imperceptibly substituted in
the outgoing headers of all MQL programs that use WebRequest for the same site.
To simplify sequential requests, it makes sense to formalize popular actions in a special class
HTTPRequest (HTTPRequest.mqh). We will store common HTTP headers in it, which are likely to be

---

## Page 1809

Part 7. Advanced language tools
1 809
7.5 Network functions
needed for all requests (for example, supported languages, instructions for proxies, etc.). In addition,
such a setting as timeout is also common. Both settings are passed to the object's constructor.
class HTTPRequest: public HttpCookie
{
protected:
   string common_headers;
   int timeout;
   
public:
   HTTPRequest(const string h, const int t = 5000):
      common_headers(h), timeout(t) { }
   ...
By default, the timeout is set to 5 seconds. The main, in a sense, universal method of the class is
request.
   int request(const string method, const string address,
      string headers, const uchar &data[], uchar &result[], string &response)
   {
      if(headers == NULL) headers = common_headers;
      
      ArrayResize(result, 0);
      response = NULL;
      Print(">>> Request:\n", method + " " + address + "\n" + headers);
      
      const int code = PRTF(WebRequest(method, address, headers, timeout, data, result, response));
      Print("<<< Response:\n", response);
      return code;
   }
};
Let's describe a couple more methods for queries of specific types.
GET requests use only headers and the body of the document (the term payload is often used) is
empty.
   int GET(const string address, uchar &result[], string &response,
      const string custom_headers = NULL)
   {
      uchar nodata[];
      return request("GET", address, custom_headers, nodata, result, response);
   }
In POST requests, there is usually a payload.
   int POST(const string address, const uchar &payload[],
      uchar &result[], string &response, const string custom_headers = NULL)
   {
      return request("POST", address, custom_headers, payload, result, response);
   }
Forms can be sent in different formats. The simplest one is "application/x-www-form-urlencoded". It
implies that the payload will be a string (maybe a very long one, since the specifications do not impose

---

## Page 1810

Part 7. Advanced language tools
1 81 0
7.5 Network functions
restrictions, and it all depends on the settings of the web servers). For such forms, we will provide a
more convenient overload of the POST method with the payload string parameter.
   int POST(const string address, const string payload,
      uchar &result[], string &response, const string custom_headers = NULL)
   {
      uchar bytes[];
      const int n = StringToCharArray(payload, bytes, 0, -1, CP_UTF8);
      ArrayResize(bytes, n - 1); // remove terminal zero
      return request("POST", address, custom_headers, bytes, result, response);
   }
Let's write a simple script to test our client web engine WebRequestCookie.mq5. Its task will be to
request the same web page twice: the first time the server will most likely offer to set its cookies, and
then they will be automatically substituted in the second request. In the input parameters, specify the
address of the page for the test: let it be the mql5.com website. We will also simulate the default
headers by the corrected "User-Agent" string.
input string Address = "https://www.mql5.com";
input string Headers = "User-Agent: Mozilla/5.0 (Windows NT 10.0) Chrome/103.0.0.0"; // Headers (use '|' as separator, if many specified)
In the main function of the script, we describe the HTTPRequest object and execute two GET requests
in a loop.
Attention! This test works under the assumption that MQL programs have not yet visited the
www.mql5.com site and have not received cookies from it. After running the script once, the
cookies will remain in the terminal cache, and it will become impossible to reproduce the example:
on both iterations of the loop, we will get the same log entries. 
Don't forget to add the "www.mql5.com" domain to the allowed list in the terminal settings.

---

## Page 1811

Part 7. Advanced language tools
1 81 1 
7.5 Network functions
void OnStart()
{
   uchar result[];
   string response;
   HTTPRequest http(Headers);
   
   for(int i = 0; i < 2; ++i)
   {
      if(http.GET(Address, result, response) > -1)
      {
         if(ArraySize(result) > 0)
         {
            PrintFormat("Got data: %d bytes", ArraySize(result));
            if(i == 0) // show the beginning of the document only the first time
            {
               const string s = CharArrayToString(result, 0, 160, CP_UTF8);
               int j = -1, k = -1;
               while((j = StringFind(s, "\r\n", j + 1)) != -1) k = j;
               Print(StringSubstr(s, 0, k));
            }
         }
      }
   }
}
The first iteration of the loop will generate the following log entries (with abbreviations):

---

## Page 1812

Part 7. Advanced language tools
1 81 2
7.5 Network functions
>>> Request:
GET https://www.mql5.com
User-Agent: Mozilla/5.0 (Windows NT 10.0) Chrome/103.0.0.0
WebRequest(method,address,headers,timeout,data,result,response)=200 / ok
<<< Response:
Server: nginx
Date: Sun, 24 Jul 2022 19:04:35 GMT
Content-Type: text/html; charset=utf-8
Transfer-Encoding: chunked
Connection: keep-alive
Cache-Control: no-cache,no-store
Content-Encoding: gzip
Expires: -1
Pragma: no-cache
Set-Cookie: sid=CfDJ8O2AwC...Ne2yP5QXpPKA2; domain=.mql5.com; path=/; samesite=lax; httponly
Vary: Accept-Encoding
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
Content-Security-Policy: default-src 'self'; script-src 'self' ... 
Generate-Time: 2823
Agent-Type: desktop-ru-en
X-Cache-Status: MISS
Got data: 184396 bytes
   
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
We received one new cookie with the name sid. To verify its effectiveness, you change to viewing the
second part of the log, for the second iteration of the loop.

---

## Page 1813

Part 7. Advanced language tools
1 81 3
7.5 Network functions
>>> Request:
GET https://www.mql5.com
User-Agent: Mozilla/5.0 (Windows NT 10.0) Chrome/103.0.0.0
WebRequest(method,address,headers,timeout,data,result,response)=200 / ok
<<< Response:
Server: nginx
Date: Sun, 24 Jul 2022 19:04:36 GMT
Content-Type: text/html; charset=utf-8
Transfer-Encoding: chunked
Connection: keep-alive
Cache-Control: no-cache, no-store, must-revalidate, no-transform
Content-Encoding: gzip
Expires: -1
Pragma: no-cache
Vary: Accept-Encoding
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
Content-Security-Policy: default-src 'self'; script-src 'self' ... 
Generate-Time: 2950
Agent-Type: desktop-ru-en
X-Cache-Status: MISS
Unfortunately, here we do not see the full outgoing headers formed inside WebRequest, but the
instance of the cookie being sent to the server using the "Cookie:" header is proven by the fact that
the server in its second response no longer asks to set it.
In theory, this cookie simply identifies the visitor (as most sites do) but does not signify their
authorization. Therefore, let's return to the exercise of submitting the form in a general way, meaning
in the future the private task of entering a login and password.
Recall that to submit the form, we can use the POST method with a string parameter payload. The
principle of preparing data according to the "x-www-form-urlencoded" standard is that named variables
and their values are written in one continuous line (somewhat similar to cookies).
name⁽№⁾=value⁽№⁾[&name⁽ⁱ⁾=value⁽ⁱ⁾...]ᵒᵖᵗ
The name and value are connected with the sign '=', and the pairs are joined using the ampersand
character '&'. The value may be missing. For example,
Name=John&Age=33&Education=&Address=
It is important to note that from a technical point of view, this string must be converted according to
the algorithm before sending urlencode (this is where the name of the format comes from), however,
WebRequest does this transformation for us.  
The variable names are determined by the web form (the contents of the tag form in a web page) or
web application logic - in any case, the web server must be able to interpret the names and values.
Therefore, to get acquainted with the technology, we need a test server with a form.
The test form is available at https://httpbin.org/forms/post. It is a dialog for ordering pizza.

---

## Page 1814

Part 7. Advanced language tools
1 81 4
7.5 Network functions
Test web form
Its internal structure and behavior are described by the following HTML code. In it, we are primarily
interested in input tags, which set the variables expected by the server. In addition, attention should be
paid to the action attribute in the form tag, since it defines the address to which the POST request
should be sent, and in this case, it is "/post", which together with the domain gives the string
"httpbin.org/post". This is what we will use in the MQL program.

---

## Page 1815

Part 7. Advanced language tools
1 81 5
7.5 Network functions
<!DOCTYPE html>
<html>
  <body>
  <form method="post" action="/post">
    <p><label>Customer name: <input name="custname"></label></p>
    <p><label>Telephone: <input type=tel name="custtel"></label></p>
    <p><label>E-mail address: <input type=email name="custemail"></label></p>
    <fieldset>
      <legend> Pizza Size </legend>
      <p><label> <input type=radio name=size value="small"> Small </label></p>
      <p><label> <input type=radio name=size value="medium"> Medium </label></p>
      <p><label> <input type=radio name=size value="large"> Large </label></p>
    </fieldset>
    <fieldset>
      <legend> Pizza Toppings </legend>
      <p><label> <input type=checkbox name="topping" value="bacon"> Bacon </label></p>
      <p><label> <input type=checkbox name="topping" value="cheese"> Extra Cheese </label></p>
      <p><label> <input type=checkbox name="topping" value="onion"> Onion </label></p>
      <p><label> <input type=checkbox name="topping" value="mushroom"> Mushroom </label></p>
    </fieldset>
    <p><label>Preferred delivery time: <input type=time min="11:00" max="21:00" step="900" name="delivery"></label></p>
    <p><label>Delivery instructions: <textarea name="comments"></textarea></label></p>
    <p><button>Submit order</button></p>
  </form>
  </body>
</html>
In the WebRequestForm.mq5 script, we have prepared similar input variables to be specified by the user
before being sent to the server.
input string Address = "https://httpbin.org/post";
   
input string Customer = "custname=Vincent Silver";
input string Telephone = "custtel=123-123-123";
input string Email = "custemail=email@address.org";
input string PizzaSize = "size=small"; // PizzaSize (small,medium,large)
input string PizzaTopping = "topping=bacon"; // PizzaTopping (bacon,cheese,onion,mushroom)
input string DeliveryTime = "delivery=";
input string Comments = "comments=";
The already set strings are shown only for one-click testing: you can replace them with your own, but
note that inside each string only the value to the right of '=' should be edited, and the name to the left
of '=' should be kept (unknown names will be ignored by the server) .
In the OnStart function, we describe the HTTP header "Content-Type:" and prepare a concatenated
string with all variables.

---

## Page 1816

Part 7. Advanced language tools
1 81 6
7.5 Network functions
void OnStart()
{
   uchar result[];
   string response;
   string header = "Content-Type: application/x-www-form-urlencoded";
   string form_fields;
   StringConcatenate(form_fields,
      Customer, "&",
      Telephone, "&",
      Email, "&",
      PizzaSize, "&",
      PizzaTopping, "&",
      DeliveryTime, "&",
      Comments);
   HTTPRequest http;
   if(http.POST(Address, form_fields, result, response) > -1)
   {
      if(ArraySize(result) > 0)
      {
         PrintFormat("Got data: %d bytes", ArraySize(result));
         // NB: UTF-8 is implied for many content-types,
 // but some may be different, analyze the response headers
         Print(CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8));
      }
   }
}
Then we execute the POST method and log the server response. Here is an example result.

---

## Page 1817

Part 7. Advanced language tools
1 81 7
7.5 Network functions
>>> Request:
POST https://httpbin.org/post
Content-Type: application/x-www-form-urlencoded
WebRequest(method,address,headers,timeout,data,result,response)=200 / ok
<<< Response:
Date: Mon, 25 Jul 2022 08:41:41 GMT
Content-Type: application/json
Content-Length: 780
Connection: keep-alive
Server: gunicorn/19.9.0
Access-Control-Allow-Origin: *
Access-Control-Allow-Credentials: true
   
Got data: 721 bytes
{
  "args": {}, 
  "data": "", 
  "files": {}, 
  "form": {
    "comments": "", 
    "custemail": "email@address.org", 
    "custname": "Vincent Silver", 
    "custtel": "123-123-123", 
    "delivery": "", 
    "size": "small", 
    "topping": "bacon"
  }, 
  "headers": {
    "Accept": "*/*", 
    "Accept-Encoding": "gzip, deflate", 
    "Accept-Language": "ru,en", 
    "Content-Length": "127", 
    "Content-Type": "application/x-www-form-urlencoded", 
    "Host": "httpbin.org", 
    "User-Agent": "MetaTrader 5 Terminal/5.3333 (Windows NT 10.0; x64)", 
    "X-Amzn-Trace-Id": "Root=1-62de5745-25bd1d823a9609f01cff04ad"
  }, 
  "json": null, 
  "url": "https://httpbin.org/post"
}
The test server acknowledges receipt of the data as a JSON copy. In practice, the server, of course,
will not return the data itself, but simply will report a success status and possibly redirect to another
web page that the data had an effect on (for example, show the order number).
With the help of such POST requests, but of smaller size, authorization is usually performed as well. But
to say the truth, most web services deliberately overcomplicate this process for security purposes and
require you to first calculate several hash sums from the user's details. Specially developed public APIs
usually have descriptions of all necessary algorithms in the documentation. But this is not always the
case. In particular, we will not be able to log in using WebRequest on mql5.com because the site does
not have an open programming interface.

---

## Page 1818

Part 7. Advanced language tools
1 81 8
7.5 Network functions
When sending requests to web services, always adhere to the rule about not exceeding the frequency of
requests: usually, each service specifies its own limits, and violation of them will lead to the subsequent
blocking of your client program, account, or IP address.
7.5.5 Establishing and breaking a network socket connection
In the previous sections, we got acquainted with the high-level MQL5 network functions: each of them
provides support for a specific application protocol. For example, SMTP is used to send emails
(SendMail), FTP is used for file transfer (SendFTP), and HTTP allows receiving web documents
(WebRequest). All the mentioned standards are based on a lower, transport layer TCP (Transmission
Control Protocol). It is not the last in the hierarchy as there are also lower ones, but we will not discuss
them here.
The standard implementation of application protocols hides many technical nuances inside and
eliminates the need for the programmer to routinely following specifications for hours. However, it does
not have flexibility and does not take into account the advanced features embedded in the standards.
Therefore, sometimes it is required to program network communication at the TCP level, that is, at the
socket level.
A socket can be viewed as analogous to a file on a disk: a socket is also described by an integer
descriptor by which data can be read or written, but this happens in a distributed network
infrastructure. Unlike files, the number of sockets on a computer is limited, and therefore the socket
descriptor must be requested from the system in advance before being associated with a network
resource (address, URL). Let's also say in advance that access to information via a socket is
streaming, that is, it is impossible to "rewind" a certain "pointer" to the beginning, as in a file.
Write and read threads do not intersect but can affect future read or write data since the transmitted
information is often interpreted by servers and client programs as control commands. Protocol
standards define if a stream contains commands or data.
The SocketCreate function allows the creation of an "empty" socket descriptor in MQL5.
int SocketCreate(uint flags = 0)
Its only parameter is reserved for the future to specify the bit pattern of the flags that determine the
mode of the socket, but at the moment only one stub flag is supported: SOCKET_DEFAULT corresponds
to the current mode and can be omitted. At the system level, this is equivalent to a socket in blocking
mode (this may be of interest to network programmers).
If successful, the function returns the socket handle. Otherwise, it returns INVALID_HANDLE.
A maximum of 1 28 sockets can be created from one MQL program. When the limit is exceeded, error
5271  (ERR_NETSOCKET_TOO_MANY_OPENED) is logged into _ LastError.
After we have opened the socket, it should be associated with a network address.
bool SocketConnect(int socket, const string server, uint port, uint timeout)
The SocketConnect function makes a socket connection to the server at the specified address and port
(for example, web servers typically run on ports 80 or 443 for HTTP and HTTPS, respectively, and
SMTP on port 25). The address can be either a domain name or an IP address.
The timeout parameter allows you to set a timeout in milliseconds to wait for a server response.

---

## Page 1819

Part 7. Advanced language tools
1 81 9
7.5 Network functions
The function returns a sign of a successful connection (true) or error (false). The error code is written
to _ LastError, for example, 5272 (ERR_NETSOCKET_CANNOT_CONNECT).
Please note that the connection address must be added to the list of allowed addresses in the
terminal settings (dialog Service -> Settings -> Advisors).
After you have finished working with the network, you should release the socket with SocketClose.
bool SocketClose(const int  socket)
The SocketClose function closes the socket by its handle, opened earlier using the SocketCreate
function. If the socket was previously connected via SocketConnect, the connection will be broken.
The function also returns an indicator of success (true) or error (false). In particular, when passing an
invalid handle to _ LastError, error 5270 (ERR_NETSOCKET_INVALIDHANDLE) is logged.
Let's remind you that all functions of this and subsequent sections are prohibited in indicators: there,
an attempt to work with sockets will result in error 401 4 (ERR_FUNCTION_NOT_ALLOWED, "The
system function is not allowed to be called").
Consider an introductory example, the SocketConnect.mq5 script. In the input parameters, you can
specify the address and port of the server. We are supposed to start testing with regular web servers
like mql5.com.
input string Server = "www.mql5.com";
input uint Port = 443;
In the function OnStart we just create a socket and bind it to a network resource.
void OnStart()
{
   PRTF(Server);
   PRTF(Port);
   const int socket = PRTF(SocketCreate());
   if(PRTF(SocketConnect(socket, Server, Port, 5000)))
   {
      PRTF(SocketClose(socket));
   }
}
If all the settings in the terminal are correct and it is connected to the Internet, we will get the
following "report".
Server=www.mql5.com / ok
Port=443 / ok
SocketCreate()=1 / ok
SocketConnect(socket,Server,Port,5000)=true / ok
SocketClose(socket)=true / ok
7.5.6 Checking socket status
When working with a socket, it becomes necessary to check its status because distributed networks
are not as reliable as a file system. In particular, the connection may be lost for one reason or another.
The SocketIsConnected function allows you to find this out.

---

## Page 1820

Part 7. Advanced language tools
1 820
7.5 Network functions
bool SocketIsConnected(const int socket)
The function checks if the socket with the specified handle (obtained from SocketCreate) is connected
to its network resource (specified in Socket Connect) and returns true in case of success.
Another function, SocketIsReadable, lets you know if there is any data to read in the system buffer
associated with the socket. This means that the computer, to which we connected at the network
address, sent (and may continue to send) data to us.
uint SocketIsReadable(const int socket)
The function returns the number of bytes that can be read from the socket. In case of error, 0 is
returned.
Programmers familiar with the Windows/Linux socket system APIs know that a value of 0 can also
be a normal state when there is no incoming data in the socket's internal buffer. However, this
function behaves differently in MQL5. With an empty system socket buffer, it speculatively returns
1 , deferring the actual check for data availability until the next call to one of the read functions. In
particular, this situation with a dummy result of 1  byte occurs, as a rule, the first time a function is
called on a socket when the receiving internal buffer is still empty.
When executing this function, an error may occur, meaning that the connection established through
SocketConnect, was broken (in _ LastError we will get code 5273, ERR_NETSOCKET_IO_ERROR).
The SocketIsReadable function is useful in programs that are designed for "non-blocking" reading of
data using SocketRead. The point is that the SocketRead function when there is no data in the receive
buffer, will wait for their arrival, suspending the execution of the program (by the specified timeout
value).
On the other hand, a blocking read is more reliable in the sense that your program will "wake up" as
soon as new data arrives, but checking for their presence with SocketIsReadable needs to be done
periodically, according to some other events (usually, on a timer or in a loop).
Particular care should be taken when using the SocketIsReadable function in TLS secure mode. The
function returns the amount of "raw" data, which in TLS mode is an encrypted block. If the "raw" data
has not yet been accumulated in the size of the decryption block, then the subsequent call of the read
function SocketTlsRead will block program execution, waiting for the missing fragment. If the "raw"
data already contains a block ready for decryption, the read function will return fewer decrypted bytes
than the number of "raw" bytes. In this regard, with TLS enabled, it is recommended to always use the
SocketIsReadable function in conjunction with SocketTlsReadAvailable. Otherwise, the behavior of the
program will differ from what is expected. Unfortunately, MQL5 does not provide the
SocketTlsIsReadable function, which is compatible with the TLS mode and does not impose the
described conventions.
The similar SocketIsWritable function checks if the given socket can be written to at the current time.
bool SocketIsWritable(const int socket)
The function returns an indication of success (true) or error (false). In the latter case, the connection
established through SocketConnect will be broken.
Here is a simple script SocketIsConnected.mq5 to test the functions. In the input parameters, we will
provide the opportunity to enter the address and port.

---

## Page 1821

Part 7. Advanced language tools
1 821 
7.5 Network functions
input string Server = "www.mql5.com";
input uint Port = 443;
In the OnStart handler, we create a socket, connect to the site, and start checking the status of the
socket in a loop. After the second iteration, we forcibly close the socket, and this should lead to an exit
from the loop.
void OnStart()
{
   PRTF(Server);
   PRTF(Port);
   const int socket = PRTF(SocketCreate());
   if(PRTF(SocketConnect(socket, Server, Port, 5000)))
   {
      int i = 0;
      while(PRTF(SocketIsConnected(socket)) && !IsStopped())
      {
         PRTF(SocketIsReadable(socket));
         PRTF(SocketIsWritable(socket));
         Sleep(1000);
         if(++i >= 2)
         {
            PRTF(SocketClose(socket));
         }
      }
   }
}
The following entries are displayed in the log.
Server=www.mql5.com / ok
Port=443 / ok
SocketCreate()=1 / ok
SocketConnect(socket,Server,Port,5000)=true / ok
SocketIsConnected(socket)=true / ok
SocketIsReadable(socket)=0 / ok
SocketIsWritable(socket)=true / ok
SocketIsConnected(socket)=true / ok
SocketIsReadable(socket)=0 / ok
SocketIsWritable(socket)=true / ok
SocketClose(socket)=true / ok
SocketIsConnected(socket)=false / NETSOCKET_INVALIDHANDLE(5270)
7.5.7 Setting data send and receive timeouts for sockets
Since network connections are unreliable, all operations with Socket functions support a centralized
timeout setting. If data reading or sending is not completed successfully within the specified time, the
function will stop trying to perform the corresponding action.
You can set timeouts for receiving and sending data using the SocketTimeouts function.

---

## Page 1822

Part 7. Advanced language tools
1 822
7.5 Network functions
bool SocketTimeouts(int socket, uint timeout_send, uint timeout_receive)
Both timeouts are given in milliseconds and affect all functions on the specified socket at the system
level.
The SocketRead function has its own timeout parameter, with which you can additionally control the
timeout during a particular call of the SocketRead function.
SocketTimeouts returns true if successful and false otherwise.
By default, there are no timeouts, which means waiting indefinitely for all data to be received or sent.
7.5.8 Reading and writing data over an insecure socket connection
Historically, sockets provide data transfer over a simple connection by default. Data transmission in an
open form allows technical means to analyze all traffic. In recent years, security issues have been
taken more seriously and therefore TLS (Transport Layer Security) technology has been implemented
almost everywhere: it provides on-the-fly encryption of all data between the sender and the recipient.
In particular, for Internet connections, the difference lies in the HTTP (simple connection) and HTTPS
(secure) protocols.
MQL5 provides different sets of Socket functions for working with simple and secure connections. In
this section, we will get acquainted with the simple mode, and later we will move to the protected one.
To read data from a socket, use the SocketRead function.
int SocketRead(int socket, uchar &buffer[], uint maxlen, uint timeout)
The socket descriptor is obtained from SocketCreate and connected to a network resource using
Socket Connect.
The buffer parameter is a reference to the array into which the data will be read. If the array is
dynamic, its size increases by the number of bytes read, but it cannot exceed INT_MAX
(21 47483647). You can limit the number of read bytes in the maxlen parameter. Data that does not fit
will remain in the socket's internal buffer: it can be obtained by the following call SocketRead. The
value of maxlen must be between 1  and INT_MAX (21 47483647).
The timeout parameter specifies the time (in milliseconds) to wait for the read to complete. If no data
is received within this time, the attempts are terminated and the function exits with the result -1 .
-1  is also returned on error, while the error code in _ LastError, for example, 5273
(ERR_NETSOCKET_IO_ERROR), means that the connection established via SocketConnect is now
broken.
If successful, the function returns the number of bytes read.
When setting the read timeout to 0, the default value of 1 20000 (2 minutes) is used.
To write data to a socket, use the SocketSend function.
Unfortunately, the function names SocketRead and SocketSend are not "symmetric": the reverse
operation for "read" is "write", and for "send" is "receive". This may be unfamiliar to developers
with experience who worked with networking APIs on other platforms.

---

## Page 1823

Part 7. Advanced language tools
1 823
7.5 Network functions
int SocketSend(int socket, const uchar &buffer[], uint maxlen)
The first parameter is a handle to a previously created and opened socket. When passing an invalid
handle, _ LastError receives error 5270 (ERR_NETSOCKET_INVALIDHANDLE). The buffer array contains
the data to be sent with the data size being specified in the maxlen parameter (the parameter was
introduced for the convenience of sending part of the data from a fixed array).
The function returns the number of bytes written to the socket on success and -1  on error.
System-level errors (5273, ERR_NETSOCKET_IO_ERROR) indicate a disconnect.
The script SocketReadWriteHTTP.mq5 demonstrates how sockets can be used to implement work over
the HTTP protocol, that is, request information about a page from a web server. This is a small part of
what the WebRequest function does for us "behind the scenes".
Let's leave the default address in the input parameters: the site "www.mql5.com". The port number
was chosen to be 80 because that is the default value for non-secure HTTP connections (although
some servers may use a different port: 81 , 8080, etc.). Ports reserved for secure connections (in
particular, the most popular 443) are not yet supported by this example. Also, in the Server parameter,
it is important to enter the name of the domain and not a specific page because the script can only
request the main page, i.e., the root path "/".
input string Server = "www.mql5.com";
input uint Port = 80;
In the main function of the script, we will create a socket and open a connection on it with the
specified parameters (the timeout is 5 seconds).
void OnStart()
{
   PRTF(Server);
   PRTF(Port);
   const int socket = PRTF(SocketCreate());
   if(PRTF(SocketConnect(socket, Server, Port, 5000)))
   {
      ...
   }
}
Let's take a look at how the HTTP protocol works. The client sends requests in the form of specially
designed headers (strings with predefined names and values), including, in particular, the web page
address, and the server sends the entire web page or operation status in response, also using special
headers for this. The client can request a web page with a GET request, send some data with a POST
request, or check the status of the web page with a frugal HEAD request. In theory, there are many
more HTTP methods – you can learn about them in the HTTP protocol specification.
Thus, the script must generate and send an HTTP header over the socket connection. In its simplest
form, the following HEAD request allows you to get meta information about the page (we could replace
HEAD with GET to request the entire page but there are some complications; we will discuss this later).

---

## Page 1824

Part 7. Advanced language tools
1 824
7.5 Network functions
HEAD / HTTP/1.1
Host: _server_
User-Agent: MetaTrader 5
                                     // <- two newlines in a row \r\n\r\n
The forward slash after "HEAD" (or another method) is the shortest possible path on any server to the
root directory, which usually results in the main page being displayed. If we wanted a specific web
page, we could write something like "GET /en/forum/ HTTP/1 .1 " and get the table of contents of the
English language forums from mql5.com. Specify a real domain instead of the "_server_" string.
Although the presence of "User-Agent:" is optional, it allows the program to "introduce itself" to the
server, without which some servers may reject the request.
Notice the two empty lines: they mark the end of the heading. In our script, it is convenient to form the
title with the following expression:
StringFormat("HEAD / HTTP/1.1\r\nHost: %s\r\n\r\n", Server)
Now we just have to send it to the server. For this purpose, we have written a simple function
HTTPSend. It receives a socket descriptor and a header line.
bool HTTPSend(int socket, const string request)
{ 
   char req[];
   int len = StringToCharArray(request, req, 0, WHOLE_ARRAY, CP_UTF8) - 1;
   if(len < 0) return false;
   return SocketSend(socket, req, len) == len;
} 
Internally, we convert the string to a byte array and call SocketSend.
Next, we need to accept the server response, for which we have written the HTTPRecv function. It also
expects a socket descriptor and a reference to a string where the data should be placed but is more
complex.

---

## Page 1825

Part 7. Advanced language tools
1 825
7.5 Network functions
bool HTTPRecv(int socket, string &result, const uint timeout)
{ 
   char response[];
   int len;         // signed integer needed for error flag -1
   uint start = GetTickCount();
   result = "";
   
   do 
   {
      ResetLastError();
      if(!(len = (int)SocketIsReadable(socket)))
      {
         Sleep(10); // wait for data or timeout
      }
      else          // read the data in the available volume
      if((len = SocketRead(socket, response, len, timeout)) > 0)
      {
         result += CharArrayToString(response, 0, len); // NB: without CP_UTF8 only 'HEAD'
         const int p = StringFind(result, "\r\n\r\n");
         if(p > 0)
         {
            // HTTP header ends with a double newline, use this
            // to make sure the entire header is received
            Print("HTTP-header found");
            StringSetLength(result, p); // cut off the body of the document (in case of a GET request)
            return true;
         }
      }
   } 
   while(GetTickCount() - start < timeout && !IsStopped() && !_LastError);
   
   if(_LastError) PRTF(_LastError);
   
   return StringLen(result) > 0;
}
Here we are checking in a loop the appearance of data within the specified timeout and reading it into
the response buffer. The occurrence of an error terminates the loop.
Buffer bytes are immediately converted to a string and concatenated into a full response in the result
variable. It is important to note that we can only use the CharArrayToString function with the default
encoding for the HTTP header because only Latin letters and a few special characters from ANSI are
allowed in it.
To receive a complete web document, which, as a rule, has UTF-8 encoding (but potentially has
another non-Latin one, which is indicated just in the HTTP header), more tricky processing will be
required: first, you need to collect all the sent blocks in one common buffer and then convert the whole
thing into a string indicating CP_UTF8 (otherwise, any character encoded in two bytes can be "cut"
when sent, and will arrive in different blocks; that is why we cannot expect a correct UTF-8 byte
stream in individual fragment). We will improve this example in the following sections.
Having functions HTTPSend and HTTPRecv, we complete the OnStart code.

---

## Page 1826

Part 7. Advanced language tools
1 826
7.5 Network functions
void OnStart()
{
      ...
      if(PRTF(HTTPSend(socket, StringFormat("HEAD / HTTP/1.1\r\nHost: %s \r\n"
         "User-Agent: MetaTrader 5\r\n\r\n", Server))))
      {
         string response;
         if(PRTF(HTTPRecv(socket, response, 5000)))
         {
            Print(response);
         }
      }
      ...
}
In the HTTP header received from the server, the following lines may be of interest:
• 'Content-Length:' – the total length of the document in bytes
• 'Content-Language:' – document language (for example, "de-DE, ru")
• 'Content-Type:' – document encoding (for example, "text/html; charset=UTF-8")
• 'Last-Modified:' – the time of the last modification of the document, so as not to download what is
already there (in principle, we can add the 'If-Modified-Since:' header in our HTTP request)
We will talk about finding out the document length (data size) in more detail because almost all headers
are optional, that is, they are reported by the server at will, and in their absence, alternative
mechanisms are used. The size is important to know when to close the connection, i.e., to make sure
that all the data has been received.
Running the script with default parameters produces the following result.
Server=www.mql5.com / ok
Port=80 / ok
SocketCreate()=1 / ok
SocketConnect(socket,Server,Port,5000)=true / ok
HTTPSend(socket,StringFormat(HEAD / HTTP/1.1
Host: %s
,Server))=true / ok
HTTP-header found
HTTPRecv(socket,response,5000)=true / ok
HTTP/1.1 301 Moved Permanently
Server: nginx
Date: Sun, 31 Jul 2022 10:24:00 GMT
Content-Type: text/html
Content-Length: 162
Connection: keep-alive
Location: https://www.mql5.com/
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
X-Frame-Options: SAMEORIGIN
Please note that this site, like most sites today, redirects our request to a secure connection: this is
achieved with the status code "301  Moved Permanently" and the new address "Location:

---

## Page 1827

Part 7. Advanced language tools
1 827
7.5 Network functions
https://www.mql5.com/" (protocol is important here " https"). To retry a TLS-enabled request, several
other functions must be used, and we will discuss them later.
7.5.9 Preparing a secure socket connection
To transfer a socket connection to a protected state and check it, MQL6 provides the following
functions: SocketTlsHandshake and SocketTlsCertificate, respectively. As a rule, we do not need to
"manually" enable protection by calling SocketTlsHandshake if the connection is established on port
443. The fact is that it is standard for HTTPS (TLS).
Protection is based on encryption of the data flow between the client and the server, for which a pair of
asymmetric keys is initially used: public and private. We have already touched on this topic in the
section Overview of available information transformation methods. Every decent site acquires a digital
certificate from one of the certification authorities (CAs) trusted by the network community. The
certificate contains the site's public key and is digitally signed by the center. Browsers and other client
applications store (or can import) the public keys of CAs and therefore can verify the quality of a
particular certificate.
Establishing a secure TLS connection 
(picture from the internet)
Further, when preparing a secure connection, the browser or application generates a certain "secret",
encrypts it with the site's public key and sends the key to it, and the site decrypts it with the private
key which only the site knows. This stage looks more complicated in practice, but as a result, both the

---

## Page 1828

Part 7. Advanced language tools
1 828
7.5 Network functions
client and the server have the encryption key for the current session (connection). This key is used by
both participants in the communication to encrypt subsequent requests and responses at one end and
decrypt them at the other.
The SocketTlsHandshake function initiates a secure TLS connection with the specified host using the
TLS handshake protocol. In this case, the client and the server agree on the connection parameters:
the version of the protocol used and the method of data encryption.
bool SocketTlsHandshake(int socket, const string host)
The socket handle and the address of the server with which the connection is established are passed in
the function parameters (in fact, this is the same name that was specified in SocketConnect).
Before a secure connection, the program must first establish a regular TCP connection with the host
using SocketConnect.
The function returns true if successful; otherwise, it returns false. In case of an error, code 5274
(ERR_NETSOCKET_HANDSHAKE_FAILED) is written in  _ LastError.
The SocketTlsCertificate function gets information about the certificate used to secure the network
connection.
int SocketTlsCertificate(int socket, string &subject, string &issuer, string &serial, string &thumbprint,
datetime &expiration)
If a secure connection is established for the socket (either after an explicit and successful
SocketTlsHandshake call or after connecting via port 443), this function fills in all other reference
variables by the socket descriptor with the corresponding information: the name of the certificate
owner (subj ect), certificate issuer name (issuer), serial number (serial), digital fingerprint (thumbprint),
and certificate validity period (expiration).
The function returns true in case of successful receipt of information about the certificate or false as a
result of an error. The error code is 5275 (ERR_NETSOCKET_NO_CERTIFICATE). This can be used to
determine whether the connection opened by the SocketConnect is immediately in protected mode. We
will use this in an example in the next section.
7.5.1 0 Reading and writing data over a secure socket connection
A secure connection has its own set of data exchange functions between the client and the server. The
names and concept of operation of the functions almost coincide with the previously considered
functions SocketRead and SocketSend.
int SocketTlsRead(int socket, uchar &buffer[], uint maxlen)
The SocketTlsRead function reads data from a secure TLS connection opened on the specified socket.
The data gets into the buffer array passed by reference. If it is dynamic, its size will be increased
according to the amount of data but no more than INT_MAX (21 47483647) bytes.
The maxlen parameter specifies the number of decrypted bytes to be received (their number is always
less than the amount of "raw" encrypted data coming into the socket's internal buffer). Data that does
not fit in the array remains in the socket and can be received by the next SocketTlsRead call.
The function is executed until it receives the specified amount of data or until the timeout specified in
SocketTimeouts occurs.

---

## Page 1829

Part 7. Advanced language tools
1 829
7.5 Network functions
In case of success, the function returns the number of bytes read; in case of error, it returns -1 , while
code 5273 (ERR_NETSOCKET_IO_ERROR) is written in _ LastError. The presence of an error indicates
that the connection was terminated.
int SocketTlsReadAvailable(int socket, uchar &buffer[], const uint maxlen)
The SocketTlsReadAvailable function reads all available decrypted data from a secure TLS connection
but no more maxlen bytes. Unlike SocketTlsRead, SocketTlsReadAvailable does not wait for the
mandatory presence of a given amount of data and immediately returns only what is present. Thus, if
the internal buffer of the socket is "empty" (nothing has been received from the server yet, it has
already been read or has not yet formed a block ready for decryption), the function will return 0 and
nothing will be recorded in the receiving array buffer. This is a regular situation.
The value of maxlen must be between 1  and INT_MAX (21 47483647).
int SocketTlsSend(int socket, const uchar &buffer[], uint bufferlen)
The SocketTlsSend function sends data from the buffer array over a secure connection opened on the
specified socket. The principle of operation is the same as that of the previously described function
SocketSend, while the only difference is in the type of connection.
Let's create a new script SocketReadWriteHTTPS.mq5 based on the previously considered
SocketReadWriteHTTP.mq5 and add flexibility in terms of choosing an HTTP method (GET by default,
not HEAD), setting a timeout, and supporting secure connections. The default port is 443.
input string Method = "GET"; // Method (HEAD,GET)
input string Server = "www.google.com";
input uint Port = 443;
input uint Timeout = 5000;
The default server is www.google.com. Do not forget to add it (and any other server that you enter) to
the list of allowed ones in the terminal settings.
To determine whether the connection is secure or not, we will use the SocketTlsCertificate function: if it
is successful, then the server has provided a certificate and TLS mode is active. If the function returns
false and throws the error code NETSOCKET_NO_CERTIFICATE(5275), this means we are using a
normal connection but the error can be ignored and reset since we are satisfied with an unsecured
connection.

---

## Page 1830

Part 7. Advanced language tools
1 830
7.5 Network functions
void OnStart()
{
   PRTF(Server);
   PRTF(Port);
   const int socket = PRTF(SocketCreate());
   if(socket == INVALID_HANDLE) return;
   SocketTimeouts(socket, Timeout, Timeout);
   if(PRTF(SocketConnect(socket, Server, Port, Timeout)))
   {
      string subject, issuer, serial, thumbprint; 
      datetime expiration;
      bool TLS = false;
      if(PRTF(SocketTlsCertificate(socket, subject, issuer, serial, thumbprint, expiration)))
      {
         PRTF(subject);
         PRTF(issuer);
         PRTF(serial);
         PRTF(thumbprint);
         PRTF(expiration);
         TLS = true;
      }
      ...
The rest of the OnStart function is implemented according to the previous plan: send a request using
the HTTPSend function and accept the answer using HTTPRecv. But this time, we additionally pass the
TLS flag to these functions, and they must be implemented slightly differently.
      if(PRTF(HTTPSend(socket, StringFormat("%s / HTTP/1.1\r\nHost: %s\r\n"
         "User-Agent: MetaTrader 5\r\n\r\n", Method, Server), TLS)))
      {
         string response;
         if(PRTF(HTTPRecv(socket, response, Timeout, TLS)))
         {
            Print("Got ", StringLen(response), " bytes");
            // for large documents, we will save to a file
            if(StringLen(response) > 1000)
            {
               int h = FileOpen(Server + ".htm", FILE_WRITE | FILE_TXT | FILE_ANSI, 0, CP_UTF8);
               FileWriteString(h, response);
               FileClose(h);
            }
            else
            {
               Print(response);
            }
         }
      }
From the example with HTTPSend, you can see that depending on the TLS flag, we use either
SocketTlsSend or SocketSend.

---

## Page 1831

Part 7. Advanced language tools
1 831 
7.5 Network functions
bool HTTPSend(int socket, const string request, const bool TLS)
{ 
   char req[];
   int len = StringToCharArray(request, req, 0, WHOLE_ARRAY, CP_UTF8) - 1;
   if(len < 0) return false;
   return (TLS ? SocketTlsSend(socket, req, len) : SocketSend(socket, req, len)) == len;
}
Things are a bit more complicated with HTTPRecv. Since we provide the ability to download the entire
page (not just the headers), we need some way to know if we have received all the data. Even after the
entire document has been transmitted, the socket is usually left open to optimize future intended
requests. But our program will not know if the transmission stopped normally, or maybe there was a
temporary "congestion" somewhere in the network infrastructure (such relaxed, intermittent page
loading can sometimes be observed in browsers). Or vice versa, in the event of a connection failure, we
may wrongly believe that we have received the entire document.
The fact is that sockets themselves act only as a means of communication between programs and work
with abstract blocks of data: they do not know the type of data, their meaning, and their logical
conclusion. All these issues are handled by application protocols like HTTP. Therefore, we will need to
delve into the specifications and implement the checks ourselves.
bool HTTPRecv(int socket, string &result, const uint timeout, const bool TLS)
{
   uchar response[]; // accumulate the data as a whole (headers + body of the web document)
   uchar block[];    // separate read block
   int len;          // current block size (signed integer for error flag -1)
   int lastLF = -1;  // position of the last line feed found LF(Line-Feed)
   int body = 0;     // offset where document body starts
   int size = 0;     // document size according to title
   result = "";      // set an empty result at the beginning
   int chunk_size = 0, chunk_start = 0, chunk_n = 1;
   const static string content_length = "Content-Length:";
   const static string crlf = "\r\n";
   const static int crlf_length = 2;
   ...
The simplest method for determining the size of the received data is based on analyzing the "Content-
Length:" header. Here we need three variables: lastLF, size, and content_ length. This header is not
always present though, and we deal with "chunks" – variables chunk_ size, chunk_ start, crlf, and
crlf_ length are introduced to detect them.
To demonstrate various techniques for receiving data, we use in this example a "non-blocking" function
SocketTlsReadAvailable. However, there is no similar function for an insecure connection, and therefore
we have to write it ourselves (a little later). The general scheme of the algorithm is simple: it is a loop
with attempts to receive new data blocks of 1 024 (or less) bytes in size. If we manage to read
something, we accumulate it in the response array. If the socket's input buffer is empty, the functions
will return 0 and we pause a little. Finally, if an error or timeout occurs, the loop will break.

---

## Page 1832

Part 7. Advanced language tools
1 832
7.5 Network functions
   uint start = GetTickCount();
   do 
   {
      ResetLastError();
      if((len = (TLS ? SocketTlsReadAvailable(socket, block, 1024) :
         SocketReadAvailable(socket, block, 1024))) > 0)
      {
         const int n = ArraySize(response);
         ArrayCopy(response, block, n); // put all the blocks together
         ...
         // main operation here
      }
      else
      {
         if(len == 0) Sleep(10); // wait a bit for the arrival of a portion of data
      }
   } 
   while(GetTickCount() - start < timeout && !IsStopped() && !_LastError);
   ...
First of all, you need to wait for the completion of the HTTP header in the input data stream. As we
have already seen from the previous example, headers are separated from the document by a double
newline, i.e., by the character sequence "\r\n\r\n". It is easy to detect by two '\n' (LF) symbols
located one after the other.
The result of the search will be the offset in bytes from the beginning of the data, where the header
ends and the document begins. We will store it in the body variable.

---

## Page 1833

Part 7. Advanced language tools
1 833
7.5 Network functions
         if(body == 0) // look for the completion of the headers until we find it
         {
            for(int i = n; i < ArraySize(response); ++i)
            {
               if(response[i] == '\n') // LF
               {
                  if(lastLF == i - crlf_length) // found sequence "\r\n\r\n"
                  {
                     body = i + 1;
                     string headers = CharArrayToString(response, 0, i);
                     Print("* HTTP-header found, header size: ", body);
                     Print(headers);
                     const int p = StringFind(headers, content_length);
                     if(p > -1)
                     {
                        size = (int)StringToInteger(StringSubstr(headers,
                           p + StringLen(content_length)));
                        Print("* ", content_length, size);
                     }
                     ...
                     break; // header/body boundary found
                  }
                  lastLF = i;
               }
            }
         }
         
         if(size == ArraySize(response) - body) // entire document
         {
            Print("* Complete document");
            break;
         }
         ...
This immediately searches for the "Content-Length:" header and extracts the size from it. The filled
size variable makes it possible to write an additional conditional statement to exit the data-receiving
loop when the entire document has been received.
Some servers give the content in parts called "chunks". In such cases, the "Transfer-Encoding:
chunked" line is present in the HTTP header, and the "Content-Length:" line is missing. Each chunk
begins with a hexadecimal number indicating the size of the chunk, followed by a newline and the
specified number of data bytes. The chunk ends with another newline. The last chunk that marks the
end of the document has a zero size.
Please note that the division into such segments is performed by the server, based on its own, current
"preferences" for optimizing sending, and has nothing to do with blocks (packets) of data into which
information is divided at the socket level for transmission over the network. In other words, chunks
tend to be arbitrarily fragmented and the boundary between network packets can even occur between
digits in a chunk size.
Schematically, this can be depicted as follows (on the left are chunks of the document, and on the right
are data blocks from the socket buffer).

---

## Page 1834

Part 7. Advanced language tools
1 834
7.5 Network functions
Fragmentation of a web document during transmission at the HTTP and TCP levels
In our algorithm, packages get  into the block array at each iteration, but it makes no sense to analyze
them one by one, and all the main work goes with the common response array.
So, if the HTTP header is completely received but the string "Content-Length:" is not found in it, we go
to the algorithm branch with the "Transfer-Encoding: chunked" mode. By the current position of body in
the response array (immediately after completion of the HTTP headers), the string fragment is
selected and converted to a number assuming the hexadecimal format: this is done by the helper

---

## Page 1835

Part 7. Advanced language tools
1 835
7.5 Network functions
function HexStringToInteger (see the attached source code). If there really is a number, we write it to
chunk_ size, mark the position as the beginning of the "chunk" in chunk_ start, and remove bytes with
the number and framing newlines from response.
                  ...
                  if(lastLF == i - crlf_length) // found sequence "\r\n\r\n"
                  {
                     body = i + 1;
                     ...
                     const int p = StringFind(headers, content_length);
                     if(p > -1)
                     {
                        size = (int)StringToInteger(StringSubstr(headers,
                           p + StringLen(content_length)));
                        Print("* ", content_length, size);
                     }
                     else
                     {
                        size = -1; // server did not provide document length
                        // try to find chunks and the size of the first one
                        if(StringFind(headers, "Transfer-Encoding: chunked") > 0)
                        {
                           // chunk syntax:
                           // <hex-size>\r\n<content>\r\n...
                           const string preview = CharArrayToString(response, body, 20);
                           chunk_size = HexStringToInteger(preview);
                           if(chunk_size > 0)
                           {
                              const int d = StringFind(preview, crlf) + crlf_length;
                              chunk_start = body;
                              Print("Chunk: ", chunk_size, " start at ", chunk_start, " -", d);
                              ArrayRemove(response, body, d);
                           }
                        }
                     }
                     break; // header/body boundary found
                  }
                  lastLF = i;
                  ...
Now, to check the completeness of the document, you need to analyze not only the size variable
(which, as we have seen, can actually be disabled by assigning -1  in the absence of "Content-Length:")
but also new variables for chunks: chunk_ start and chunk_ size. The scheme of action is the same as
after the HTTP headers: by offset in the response array, where the previous chunk ended, we isolate
the size of the next "chunk". We continue the process until we find a chunk of size zero.

---

## Page 1836

Part 7. Advanced language tools
1 836
7.5 Network functions
         ...
         if(size == ArraySize(response) - body) // entire document
         {
            Print("* Complete document");
            break;
         }
         else if(chunk_size > 0 && ArraySize(response) - chunk_start >= chunk_size)
         {
            Print("* ", chunk_n, " chunk done: ", chunk_size, " total: ", ArraySize(response));
            const int p = chunk_start + chunk_size;
            const string preview = CharArrayToString(response, p, 20);
            if(StringLen(preview) > crlf_length              // there is '\r\n...\r\n' ?
               && StringFind(preview, crlf, crlf_length) > crlf_length)
            {
               chunk_size = HexStringToInteger(preview, crlf_length);
               if(chunk_size > 0)
               {                              // twice '\r\n': before and after chunk size
                  int d = StringFind(preview, crlf, crlf_length) + crlf_length;
                  chunk_start = p;
                  Print("Chunk: ", chunk_size, " start at ", chunk_start, " -", d);
                  ArrayRemove(response, chunk_start, d);
                  ++chunk_n;
               }
               else
               {
                  Print("* Final chunk");
                  ArrayRemove(response, p, 5); // "\r\n0\r\n"
                  break;
               }
            } // otherwise wait for more data
         }
Thus, we provided an exit from the loop based on the results of the analysis of the incoming stream in
two different ways (in addition to exiting by timeout and by error). At the regular end of the loop, we
convert that part of the array into the response string, which starts from the body position and contains
the whole document. Otherwise, we simply return everything that we managed to get, along with the
headers, for "analysis".

---

## Page 1837

Part 7. Advanced language tools
1 837
7.5 Network functions
bool HTTPRecv(int socket, string &result, const uint timeout, const bool TLS)
{
   ...
   do 
   {
      ResetLastError();
      if((len = (TLS ? SocketTlsReadAvailable(socket, block, 1024) :
         SocketReadAvailable(socket, block, 1024))) > 0)
      {
         ... // main operation here - discussed above
      }
      else
      {
         if(len == 0) Sleep(10); // wait a bit for the arrival of a portion of data
      }
   } 
   while(GetTickCount() - start < timeout && !IsStopped() && !_LastError);
      
   if(_LastError) PRTF(_LastError);
   
   if(ArraySize(response) > 0)
   {
      if(body != 0)
      {
         // TODO: Desirable to check 'Content-Type:' for 'charset=UTF-8'
         result = CharArrayToString(response, body, WHOLE_ARRAY, CP_UTF8);
      }
      else
      {
         // to analyze wrong cases, return incomplete headers as is
         result = CharArrayToString(response);
      }
   }
   
   return StringLen(result) > 0;
}
The only remaining function is SocketReadAvailable which is the analog of SocketTlsReadAvailable for
unsecured connections.
int SocketReadAvailable(int socket, uchar &block[], const uint maxlen = INT_MAX)
{
   ArrayResize(block, 0);
   const uint len = SocketIsReadable(socket);
   if(len > 0)
      return SocketRead(socket, block, fmin(len, maxlen), 10);
   return 0;
}
The script is ready for work.
It took us quite a bit of effort to implement a simple web page request using sockets. This serves as a
demonstration of how much of a chore is usually hidden in the support of network protocols at a low

---

## Page 1838

Part 7. Advanced language tools
1 838
7.5 Network functions
level. Of course, in the case of HTTP, it is easier and more correct for us to use the built-in
implementation of WebRequest, but it does not include all the features of HTTP (moreover, we touched
on HTTP 1 .1  in passing, but there is also HTTP / 2), and the number of other application protocols is
huge. Therefore, Socket functions are required to integrate them in MetaTrader 5.
Let's run SocketReadWriteHTTPS.mq5 with default settings.
Server=www.google.com / ok
Port=443 / ok
SocketCreate()=1 / ok
SocketConnect(socket,Server,Port,Timeout)=true / ok
SocketTlsCertificate(socket,subject,issuer,serial,thumbprint,expiration)=true / ok
subject=CN=www.google.com / ok
issuer=C=US, O=Google Trust Services LLC, CN=GTS CA 1C3 / ok
serial=00c9c57583d70aa05d12161cde9ee32578 / ok
thumbprint=1EEE9A574CC92773EF948B50E79703F1B55556BF / ok
expiration=2022.10.03 08:25:10 / ok
HTTPSend(socket,StringFormat(%s / HTTP/1.1
Host: %s
,Method,Server),TLS)=true / ok
* HTTP-header found, header size: 1080
HTTP/1.1 200 OK
Date: Mon, 01 Aug 2022 20:48:35 GMT
Expires: -1
Cache-Control: private, max-age=0
Content-Type: text/html; charset=ISO-8859-1
Server: gws
X-XSS-Protection: 0
X-Frame-Options: SAMEORIGIN
Set-Cookie: 1P_JAR=2022-08-01-20; expires=Wed, 31-Aug-2022 20:48:35 GMT;
   path=/; domain=.google.com; Secure
...
Accept-Ranges: none
Vary: Accept-Encoding
Transfer-Encoding: chunked
Chunk: 22172 start at 1080 -6
* 1 chunk done: 22172 total: 24081
Chunk: 30824 start at 23252 -8
* 2 chunk done: 30824 total: 54083
* Final chunk
HTTPRecv(socket,response,Timeout,TLS)=true / ok
Got 52998 bytes
As we can see, the document is transferred in chunks and has been saved to a temporary file (you can
find it in MQL5/Files/www.mql5.com.htm).
Let's now run the script for the site "www.mql5.com" and port 80. From the previous section, we know
that the site in this case issues a redirect to its protected version but this "redirect" is not empty: it
has a stub document, and now we can get it in full. What matters to us here is that the "Content-
Length:" header is used correctly in this case.

---

## Page 1839

Part 7. Advanced language tools
1 839
7.5 Network functions
Server=www.mql5.com / ok
Port=80 / ok
SocketCreate()=1 / ok
SocketConnect(socket,Server,Port,Timeout)=true / ok
HTTPSend(socket,StringFormat(%s / HTTP/1.1
Host: %s
,Method,Server),TLS)=true / NETSOCKET_NO_CERTIFICATE(5275)
* HTTP-header found, header size: 291
HTTP/1.1 301 Moved Permanently
Server: nginx
Date: Sun, 31 Jul 2022 19:28:57 GMT
Content-Type: text/html
Content-Length: 162
Connection: keep-alive
Location: https://www.mql5.com/
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
X-Frame-Options: SAMEORIGIN
* Content-Length:162
* Complete document
HTTPRecv(socket,response,Timeout,TLS)=true / ok
<html>
<head><title>301 Moved Permanently</title></head>
<body>
<center><h1>301 Moved Permanently</h1></center>
<hr><center>nginx</center>
</body>
</html>
Another, large example of the use of sockets in practice, we will consider in the chapter Projects.
7.6 SQLite database
MetaTrader 5 provides native support for the SQLite database. It is a light yet fully functional database
management system (DBMS). Traditionally, such systems are focused on processing data tables, where
records of the same type are stored with a common set of attributes, and different correspondences
(links or relations) can be established between records of different types (i.e. tables), and therefore
such databases are also called relational. We have already considered examples of such connections
between structures of the economic calendar, but the calendar database is stored inside the terminal,
and the functions of this section will allow you to create arbitrary databases from MQL programs.
The specialization of the DBMS on these data structures allows you to optimize – speed up and simplify
– many popular operations such as sorting, searching, filtering, summing up, or calculating other
aggregate functions for large amounts of data.
However, there is another side to this: DBMS programming requires its own SQL (Structured Query
Language), and knowledge of pure MQL5 will not be enough. Unlike MQL5, which refers to imperative
languages (those using operators indicating what, how, in what sequence to do), SQL is declarative,
that is, it describes the initial data and the desired result, without specifying how and in what sequence
to perform calculations. The meaning of the algorithm in SQL is described in the form of SQL queries. A
query is an analog of a separate MQL5 operator, formed as a string using a special syntax.

---

## Page 1840

Part 7. Advanced language tools
1 840
7.6 SQLite database
Instead of programming complex loops and comparisons, we can simply call SQLite functions (for
example, DatabaseExecute or Database Prepare) by passing SQL queries to them. To get query results
into a ready-made MQL5 structure, you can use the DatabaseReadBind function. This will allow you to
read all the fields of the record (structure) at once in one call.
With the help of database functions, it is easy to create tables, add records to them, make
modifications, and make selections according to complex conditions, for example, for tasks such as:
• Obtaining trading history and quotes
• Saving optimization and testing results
• Preparing and exchanging data with other analysis packages
• Analyzing economic calendar data
• Storing settings and states of MQL5 programs
In addition, a wide range of common, statistical, and mathematical functions can be used in SQL
queries. Moreover, expressions with their participation can be calculated even without creating a table.
SQLite does not require a separate application, configuration, and administration, is not resource-
demanding, and supports most commands of the popular SQL92 standard. An added convenience is
that the entire database resides in a single file on the hard drive on the user's computer and can be
easily transferred or backed up. However, to speed up read, write, and modification operations, the
database can also be opened/created in RAM with the flag DATABASE_OPEN_MEMORY, however, in this
case, such a database will be available only to this particular program and cannot be used for joint work
of several programs.
It is important to note that the relative simplicity of SQLite, compared to full-featured DBMSs,
comes with some limitations. In particular, SQLite does not have a dedicated process (system
service or application) that would provide centralized access to the database and table
management API, which is why parallel, shared access to the same database (file) from different
processes is not guaranteed. So, if you need to simultaneously read and write to the database from
optimization agents that execute instances of the same Expert Advisor, you will need to write code
in it to synchronize access (otherwise, the data being written and read will be in an inconsistent
state: after all, the order of writing, modifying, deleting, and reading from concurrent
unsynchronized processes are random). Moreover, attempts to change the database at the same
time may result in the MQL program receiving "database busy" errors (and the requested operation
is not performed). The only scenario that does not require synchronization of parallel operations
with SQLite is when only read operations are involved.
We will present only the basics of SQL to the extent necessary to start applying it. A complete
description of the syntax and how SQL works is beyond the scope of this book. Check out the
documentation on the SQLite site. However, please note that MQL5 and MetaEditor support a limited
subset of commands and SQL syntax constructions.
MQL Wizard in MetaEditor has an embedded option to create a database, which immediately offers to
create the first table by defining a list of its fields. Also, the Navigator provides a separate tab for
working with databases.
Using the Wizard or the context menu of the Navigator, you can create an empty database (a file on
disk, placed by default, in the directory MQL5/Files) of supported formats (*.db, *.sql, *.sqlite and
others). In addition, in the context menu, you can import the entire database from an sql file or
individual tables from csv files.

---

## Page 1841

Part 7. Advanced language tools
1 841 
7.6 SQLite database
An existing or created database can be easily opened through the same menu. After that, its tables will
appear in the Navigator, and the right, main area of the window will display a panel with tools for
debugging SQL queries and a table with the results. For example, double-clicking on a table name
performs a quick query of all record fields, which corresponds to the "SELECT * FROM 'table'"
statement that appears in the input field at the top.
Viewing SQLite Database in MetaEditor
You can edit the request and click the Execute button to activate it. Potential SQL syntax errors are
output in the log.
For further details about the Wizard, the import/export of databases, and the interactive work with
them, please see MetaEditor documentation.
7.6.0 Principles of database operations in MQL5
Databases store information in the form of tables. Getting, modifying, and adding new data to them is
done using queries in the SQL language. We will describe its specifics in the following sections. In the
meantime, let's use the DatabaseRead.mq5 script, which has nothing to do with trading, and see how to
create a simple database and get information from it. All functions mentioned here will be described in
detail later. Now it is important to imagine the general principles.
Creating and closing a database using built-in DatabaseOpen/DatabaseClose functions are similar to
working with files as we also create a descriptor for the database, check it, and close it at the end.

---

## Page 1842

Part 7. Advanced language tools
1 842
7.6 SQLite database
void OnStart()
{
   string filename = "company.sqlite";
   // create or open a database
   int db = DatabaseOpen(filename, DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE);
   if(db == INVALID_HANDLE)
   {
      Print("DB: ", filename, " open failed with code ", _LastError);
      return;
   }
   ...// further work with the database
   // close the database
   DatabaseClose(db);
}
After opening the database, we will make sure that there is no table in it under the name we need. If
the table already exists, then when trying to insert the same data into it as in our example, an error will
occur, so we use the DatabaseTableExists function.
Deleting and creating a table is done using queries that are sent to the database with two calls to the
DatabaseExecute function and accompanied by error checking.
   ...
   // if the table COMPANY exists, then delete it
   if(DatabaseTableExists(db, "COMPANY"))
   {
      if(!DatabaseExecute(db, "DROP TABLE COMPANY"))
      {
         Print("Failed to drop table COMPANY with code ", _LastError);
         DatabaseClose(db);
         return;
      }
   }
   // creating table COMPANY 
   if(!DatabaseExecute(db, "CREATE TABLE COMPANY("
     "ID      INT     PRIMARY KEY NOT NULL,"
     "NAME    TEXT    NOT NULL,"
     "AGE     INT     NOT NULL,"
     "ADDRESS CHAR(50),"
     "SALARY  REAL );"))
   {
      Print("DB: ", filename, " create table failed with code ", _LastError);
      DatabaseClose(db);
      return;
   }
   ...
Let's explain the essence of SQL queries. In the COMPANY table, we have only 5 fields: record ID,
name, age, address, and salary. Here the ID field is a key, that is, a unique index. Indexes allow each
record to be uniquely identified and can be used across tables to link them together. This is similar to
how the position ID links all trades and orders that belong to a particular position.
Now you need to fill the table with data, this is done using the "INSERT" query:

---

## Page 1843

Part 7. Advanced language tools
1 843
7.6 SQLite database
   // insert data into table
   if(!DatabaseExecute(db,
      "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (1,'Paul',32,'California',25000.00); "
      "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (2,'Allen',25,'Texas',15000.00); "
      "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (3,'Teddy',23,'Norway',20000.00);"
      "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (4,'Mark',25,'Rich-Mond',65000.00);"))
   {
      Print("DB: ", filename, " insert failed with code ", _LastError);
      DatabaseClose(db);
      return;
   }
   ...
Here, 4 records are added to the table COMPANY, for each record there is a list of fields, and values
that will be written to these fields are indicated. Records are inserted by separate "INSERT..." queries,
which are combined into one line, through a special delimiter character ';', but we could insert each
record into the table with a separate DatabaseExecute call.
Since at the end of the script the database will be saved to the "company.sqlite" file, the next time it is
run, we would try to write the same data to the COMPANY table with the same ID. This would lead to
an error, which is why we previously deleted the table so that we would start from scratch every time
the script was run.
Now we get all records from the COMPANY table with the field SALARY > 1 5000. This is done using the
DatabasePrepare function, which "compiles" the request text and returns its handle for later use in the
DatabaseRead or DatabaseReadBind functions.
   // prepare a request with a descriptor
   int request = DatabasePrepare(db, "SELECT * FROM COMPANY WHERE SALARY>15000");
   if(request == INVALID_HANDLE)
   {
      Print("DB: ", filename, " request failed with code ", _LastError);
      DatabaseClose(db);
      return;
   }
   ...
After the request has been successfully created, we need to get the results of its execution. This can
be done using the DatabaseRead function, which on the first call will execute the query and jump to the
first record in the results. On each subsequent call, it will read the next record until it reaches the end.
In this case, it will return false, which means "there are no more records".

---

## Page 1844

Part 7. Advanced language tools
1 844
7.6 SQLite database
   // printing all records with salary over 15000
   int id, age;
   string name, address;
   double salary;
   Print("Persons with salary > 15000:");
   for(int i = 0; DatabaseRead(request); i++)
   {
      // read the values of each field from the received record by its number
      if(DatabaseColumnInteger(request, 0, id) && DatabaseColumnText(request, 1, name) &&
         DatabaseColumnInteger(request, 2, age) && DatabaseColumnText(request, 3, address) &&
         DatabaseColumnDouble(request, 4, salary))
         Print(i, ":  ", id, " ", name, " ", age, " ", address, " ", salary);
      else
      {
         Print(i, ": DatabaseRead() failed with code ", _LastError);
         DatabaseFinalize(request);
         DatabaseClose(db);
         return;
      }
   }
   // deleting handle after use
   DatabaseFinalize(request);
The result of execution will be:
Persons with salary > 15000:
0:  1 Paul 32 California 25000.0
1:  3 Teddy 23 Norway 20000.0
2:  4 Mark 25 Rich-Mond  65000.0
The DatabaseRead function allows you to go through all the records from the query result and then get
complete information about each column in the resulting table via DatabaseColumn functions. These
functions are designed to work universally with the results of any query but the cost is a redundant
code.
If the structure of the query results is known in advance, it is better to use the DatabaseReadBind
function, which allows you to read the entire record at once into a structure. We can remake the
previous example in this way and present it under a new name DatabaseReadBind.mq5. Let's first
declare the Person structure:
struct Person
{
   int    id;
   string name;
   int    age;
   string address;
   double salary;
};
Then we will subtract each record from the query results with DatabaseReadBind(request, person) in a
loop as long as the function returns true:

---

## Page 1845

Part 7. Advanced language tools
1 845
7.6 SQLite database
   Person person;
   Print("Persons with salary > 15000:");
   for(int i = 0; DatabaseReadBind(request, person); i++)
      Print(i, ":  ", person.id, " ", person.name, " ", person.age,
         " ", person.address, " ", person.salary);
   DatabaseFinalize(request);
Thus, we immediately get the values of all fields from the current record and we do not need to read
them separately.
This introductory example was taken from the article SQLite: native work with SQL databases in MQL5,
where, in addition to it, several options for the application of the database for traders are considered.
Specifically, you can find there restoring the history of positions from trades, analyzing a trading report
in terms of strategies, working symbols, or the most preferred trading hours, as well as techniques for
working with optimization results.
Some basic knowledge of SQL may be required to master this material, so we will cover it briefly in the
following sections.
7.6.1  SQL Basics
All tasks performed in SQLite assume the presence of a working database (one or more), so creating
and opening a database (similar to a file) are mandatory framework operations that establish the
necessary programming environment. There is no facility for programmatic deletion of the database in
SQLite as it is assumed that you can simply delete the database file from disk.
The actions available in the context of an open base can be conditionally divided into the following main
groups:
• Creating and deleting tables, as well as modifying their schemas, i.e., column descriptions, including
the identification of types, names, and restrictions
• Creating (adding), reading, editing, and deleting records in tables; these operations are often
denoted by the common abbreviation CRUD (Create, Read, Update, Delete)
• Building queries to select records from one or a combination of several tables according to complex
conditions
• Optimizing algorithms by building indexes on selected columns, using views (view), wrapping batch
actions in transactions, declaring event processing triggers, and other advanced tools
In SQL databases, all of these actions are performed using reserved SQL commands (or statements).
Due to the specifics of integration with MQL5, some of the actions are performed by built-in MQL5
functions. For example, opening, applying, or canceling a transaction is performed by the trinity of
DatabaseTransaction functions, although the SQL standard (and the public implementation of SQLite)
has corresponding SQL commands (BEGIN TRANSACTION, COMMIT, and ROLLBACK).
Most SQL commands are also available in MQL programs: they are passed to the SQLite executing
engine as string parameters of the DatabaseExecute or DatabasePrepare functions. The difference
between these two options lies in several nuances.
DatabasePrepare allows you to prepare a query for its subsequent mass cyclic execution with different
parameter values at each iteration (the parameters themselves, that is, their names in the query, are
the same). In addition, these prepared queries provide a mechanism to read the results using

---

## Page 1846

Part 7. Advanced language tools
1 846
7.6 SQLite database
DatabaseRead and DatabaseReadBind. So, you can use them for operations with a set of selected
records.
In contrast, the DatabaseExecute function executes the passed single query unilaterally: the command
goes inside the SQLite engine, performs some actions on the data, but returns nothing. This is
commonly used for table creation or batch modification of data.
In the future, we will often have to operate with several basic concepts. Let's introduce them:
Table – a structured set of data, consisting of rows and columns. Each row is a separate data record
with fields (properties) described using the name and type of the corresponding columns. All database
tables are physically stored in the database file and are available for reading and writing (if rights were
not restricted when opening the database).
View – a kind of virtual table calculated by the SQLite engine based on a given SQL query, other tables,
or views. Views are read-only. Unlike any tables (including temporary ones that SQL allows you to
create in memory for the duration of a program session), views are dynamically recalculated each time
they are accessed.
Index – a service data structure (the balanced tree, B-tree) for quick search of records by the values
of predefined fields (properties) or their combinations.
Trigger – a subroutine of one or more SQL statements assigned to be automatically run in response to
events (before or after) adding, changing, or deleting a record in a particular table.
Here is a short list of the most popular SQL statements and the actions they perform:
• CREATE – creates a database object (table, view, index, trigger);
• ALTER – changes an object (table);
• DROP – deletes an object (table, view, index, trigger);
• SELECT – selects records or calculates values that satisfy the given conditions;
• INSERT – adds new data (one or a set of records);
• UPDATE – changes existing records;
• DELETE – deletes records from the table;
The list only shows the keywords that start the corresponding SQL language construct. A more detailed
syntax will be shown below. Their practical application will be shown in the following examples.
Each statement can span multiple lines (linefeed characters and extra spaces are ignored). If
necessary, you can send several commands to SQLite at once. In this case, after each command, you
should use the command termination character ';' (semicolon).
The text in commands is analyzed by the system regardless of case, but in SQL it is customary to write
keywords in capital letters.
When creating a table, we must specify its name, as well as a list of columns in parentheses, separated
by commas. Each column is given a name, a type, and optionally a constraint. The simplest form:
CREATE TABLE table_name
  ( column_name type [ constraints ] [, column_name type [ constraints ...] ...]);
We will see the restrictions in SQL in the next section. In the meantime, let's have a look at a clear
example (with different types and options):

---

## Page 1847

Part 7. Advanced language tools
1 847
7.6 SQLite database
CREATE TABLE IF NOT EXISTS example_table
   (id INTEGER PRIMARY KEY,
    name TEXT,
    timestamp INTEGER DEFAULT CURRENT_STAMP,
    income REAL,
    data BLOB);
The syntax for creating an index is:
CREATE [ UNIQUE ] INDEX index_name
  ON table_name( column_name [, column_name ...]);
Existing indexes are automatically used in queries with filter conditions on the corresponding columns.
Without indexes, the process is slower.
Deleting a table (along with the data, if something has been written to it) is quite simple:
DROP TABLE table_name;
You can insert data into a table like this:
INSERT INTO table_name [ ( column_name [, column_name ...] ) ]
  VALUES( value [, value ...]);
The first list in parentheses includes the column names and is optional (see explanation below). It must
match the second list with values for them. For example,
INSERT INTO example_table (name, income) VALUES ('Morning Flat Breakout', 1000);
Note that string literals are enclosed in single quotes in SQL.
If the column names are omitted from the INSERT statement, the VALUES keyword is assumed to be
followed by the values for all the columns in the table, and in the exact order in which they are
described in the table.
There are also more complex forms of the operator, allowing, in particular, the insertion of records from
other tables or query results.
Selecting records by condition, with an optional limitation of the list of returned fields (columns), is
performed by the SELECT command.
SELECT column_name [, column_name ...] FROM table_name [WHERE condition ];
If you want to return every matching record in its entirety (all columns), use the star notation:
SELECT *FROM table_name [WHERE condition ];
When the condition is not present, the system returns all records in the table.
As a condition, you can substitute a logical expression that includes column names and various
comparison operators, as well as built-in SQL functions and the results of a nested SELECT query (such
queries are written in parentheses). Comparison operators include:
• Logical AND
• Logical OR
• IN for a value from the list
• NOT IN  for a value outside the list

---

## Page 1848

Part 7. Advanced language tools
1 848
7.6 SQLite database
• BETWEEN  for a value in the range
• LIKE – similar in spelling to a pattern with special wildcard characters ('%', '_')
• EXISTS – check for non-emptiness of the results of the nested query
For example, a selection of record names with an income of at least 1 000 and no older than one year
(preliminarily rounded to the nearest month):
SELECT name FROM example_table
  WHERE income >= 1000 AND timestamp > datetime('now', 'start of month', '-1 year');
Additionally, the selection can be sorted in ascending or descending order (ORDER BY), grouped by
characteristics (GROUP BY), and filtered by groups (HAVING). We can also limit the number of records
in it (LIMIT, OFFSET). For each group, you can return the value of any aggregate function, in
particular, COUNT, SUM, MIN, MAX, and AVG, calculated on all group records.
SELECT [ DISTINCT ] column_name [, column_name...](i) FROM table_name
  [ WHERE condition ]
  [ORDER BY column_name [ ASC | DESC ]
     [ LIMIT quantity OFFSET start_offset ] ]
  [ GROUP BY column_name ⌠ HAVING condition ] ];
The optional keyword DISTINCT allows you to remove duplicates (if they are found in the results
according to the current selection criteria). It only makes sense in the absence of grouping.
LIMIT will only give reproducible results if sorting is present.
If necessary, the SELECT selection can be made not from one table but from several, combining them
according to the required combination of fields. The keyword JOIN is used for this.
SELECT [...] FROM table name_1
  [ INNER | OUTER | CROSS ] JOIN table_name_2
  ON boolean_condition
or
SELECT [...] FROM table name_1
  [ INNER | OUTER | CROSS ] JOIN table_name_2
  USING ( common_column_name [, common_column_name ...] )
SQLite supports three kinds of JOINs: INNER JOIN, OUTER JOIN, and CROSS JOIN. The book provides
a general idea of them from examples, while you can further explore the details on your own.
For example, using JOIN, you can build all combinations of records from one table with records from
another table or compare deals from the deals table (let's call it "deals") with deals from the same
table according to the principle of matching position identifiers, but in such a way that the direction of
deals (entry to the market/exit from the market) was the opposite, resulting in a virtual table of trades.

---

## Page 1849

Part 7. Advanced language tools
1 849
7.6 SQLite database
SELECT // list the columns of the results table with aliases (after 'as')
  d1.time as time_in, d1.position_id as position, d1.type as type, // table d1
   d1.volume as volume, d1.symbol as symbol, d1.price as price_in,
  d2.time as time_out, d2.price as price_out,                      // table d2
   d2.swap as swap, d2.profit as profit,
  d1.commission + d2.commission as commission                      // combination
  FROM deals d1 INNER JOIN deals d2      // d1 and d2 - aliases of one table "deals"
  ON d1.position_id = d2.position_id     // merge condition by position
  WHERE d1.entry = 0 AND d2.entry = 1    // selection condition "entry/exit"
This is an SQL query from the MQL5 help, where JOIN examples are available in descriptions of the
DatabaseExecute and DatabasePrepare functions.
The fundamental property of SELECT is that it always returns results to the calling program, unlike
other queries such as CREATE, INSERT, etc. However, starting from SQLite 3.35, INSERT, UPDATE,
and DELETE statements also have the ability to return values, if necessary, using the additional
RETURNING keyword. For example,
INSERT INTO example_table (name, income) VALUES ('Morning Flat Breakout', 1000)
   RETURNING id;
In any case, query results in MQL5 are accessed through DatabaseColumn functions, DatabaseRead,
and DatabaseReadBind.  
In addition, SELECT allows you to evaluate the results of expressions and return them as they are or
combine them with results from tables. Expressions can include most of the operators we are familiar
with from MQL5 expressions, as well as built-in SQL functions. See the SQLite documentation for a
complete list. For example, here's how you can find the current build version of SQLite in your terminal
and editor instance, which can be important for finding out which options are available.
SELECT sqlite_version();
Here the entire expression consists of a single call of the sqlite_ version function. Similar to selecting
multiple columns from a table, you can evaluate multiple expressions separated by commas.
Several popular statistical and mathematical functions are also available.
Records should be edited with an UPDATE statement.
UPDATE table_name SET column_name = value [, column_name = value ...] 
  WHERE condition;
The syntax for the deletion command is as follows:
DELETE FROM table_name WHERE condition;
7.6.2 Structure of tables: data types and restrictions
When describing table fields, you need to specify data types for them, but the concept of a data type in
SQLite is very different from MQL5.
MQL5 is a strongly typed language: each variable or structure field always retains the data type
according to the declaration. SQL, on the other hand, is a loosely typed language: the types that we
specify in the table description are nothing more than a recommendation. The program can write a

---

## Page 1850

Part 7. Advanced language tools
1 850
7.6 SQLite database
value of an arbitrary type to any "cell" (a field in the record), and the "cell" will change its type, which,
in particular, can be detected by the built-in MQL function DatabaseColumnType.
Of course, in practice, most users tend to stick to "respect" column types.
The second significant difference in the SQL type mechanism is the presence of a large number of
keywords that describe types, but all these words ultimately come down to five storage classes. Being
a simplified version of SQL, SQLite in most cases does not distinguish between keywords of the same
group (for example, in the description of a string with a VARCHAR(80) length limit, this limit is not
controlled, and the description is equivalent to the TEXT storage class), so it is more logical to describe
the type by the group name. Specific types are left only for compatibility with other DBMS (but this is
not important for us).
The following table lists the MQL5 types and their corresponding "affinities" (which mean generalizing
features of SQL types).
MQL5 types
Generic SQL types
NULL (not a type in MQL5)
NULL (no value)
bool, char, short, int, long, uchar, ushort,
uint, ulong, datetime, color, enum
INTEGER
float, double
REAL
(real number of fixed precision, 
no analog in MQL5)
NUMERIC
string
TEXT
(arbitrary "raw" data, 
analog of uchar[] array or others)
BLOB (binary large object), NONE
When writing a value to the SQL database, it determines its type according to several rules:
• The absence of quotes, decimal point, or exponent give INTEGER
• The presence of a decimal point and an exponent means REAL
• framing of single or double quotes signals the TEXT type
• a NULL value without quotes corresponds to the NULL class
• literals (constants) with binary data are written as a hexadecimal string prefixed with 'x'
Special SQL function typeof allows you to check the type of a value. For example, the following query
can be run in the MetaEditor.
SELECT typeof(100), typeof(10.0), typeof('100'), typeof(x'1000'), typeof(NULL);
It will output to the results table:
integer | real | text | blob | null
You cannot check values for NULL by comparing '=' (because the result will also give NULL), you
should use the special NOT NULL operator.

---

## Page 1851

Part 7. Advanced language tools
1 851 
7.6 SQLite database
SQLite imposes some limits on stored data: some of them are difficult to achieve (and therefore we will
omit them here), but others can be taken into account when designing a program. So, the maximum
number of columns in the table is 2000, and the size of one row, BLOB, and in general one record
cannot exceed one million bytes. The same value is chosen as the SQL query length limit.
As far as dates and times are concerned, SQL can in theory store them in three formats, but only the
first one matches datetime in MQL5:
• INTEGER – the number of seconds since 1 970.01 .01  (also known as the "Unix epoch")
• REAL – the number of days (with fractions) from November 24, 471 4 BC
• TEXT – date and time with accuracy to the millisecond in the format "YYYY-MM-DD
HH:mm:SS.sss", optionally with the time zone, for which the suffix "[±]HH:mm" is added with an
offset from UTC
A real date storage type (also called the Julian day, for which there is a built-in SQL function Julianday)
is interesting in that it allows you to store time accurate to milliseconds. In theory, this can also be
done as a 'YYYY-MM-DDTHH:mm:SS.sssZ' format string, but such storage is very uneconomical. The
conversion of the "day" into the number of seconds with a fractional part, starting from the familiar
date 1 970.01 .01  00:00:00, is made according to the formula: j ulianday(' now' ) - 2440587.5) * 86400.0.
'Now' here denotes the current UTC time but can be changed to other values described in the SQLite
documentation. The constant 2440587.5 is exactly equal to the number of "calendar" days for the
specified "zero" date – the starting point of the "Unix epoch".
In addition to the type, each field can have one or more constraints, which are written with special
keywords after the type. A constraint describes what values the field can take and even allows you to
automate the completion according to the field's predefined purpose.
Let's consider the main constraints.
... DEFAULT expression
When adding a new record, if the field value is not specified, the system will automatically enter the
value (constant) specified here or calculate the expression (function).
... CHECK ( boolean_expression )
When adding a new record, the system will check that the expression, which can contain field names as
variables, is true. If the expression is false, the record will not be inserted and the system will return an
error.
... UNIQUE
The system checks that all records in the table have different values for this field. Attempting to add an
entry with a value that already exists will result in an error and the addition will not occur.
To track uniqueness, the system implicitly creates an index for the specified field.
... PRIMARY KEY
A field marked with this attribute is used by the system to identify records in a table and links to them
from other tables (this is how relational relationships are formed, giving the name to relational
databases in question like SQLite). Obviously, this feature also includes a unique index.
If the table does not have an INTEGER type field with the PRIMARY KEY attribute, the system
automatically implicitly creates such a column named rowid. If your table has an integer field declared
as a primary key, then it is also available under the alias rowid.

---

## Page 1852

Part 7. Advanced language tools
1 852
7.6 SQLite database
If a record with an omitted or NULL rowid is added to the table, SQLite will automatically assign it the
next integer (64-bit, corresponding to long in MQL5), larger than the maximum rowid in the table by 1 .
The initial value is 1 .
Usually the counter just increments by 1  each time, but if the number of records ever inserted into one
table (and possibly then deleted) exceeds long, the counter will jump to the beginning and the system
will try to find free numbers. But this is unlikely. For example, if you write ticks to a table at an average
rate of 1  tick per millisecond, then the overflow will occur in 292 million years.
There can be only one primary key, but it can consist of several columns, which is done using a syntax
other than constraints directly in the table description.
CREATE TABLE table_name (
  column_name type [ restrictions ]
  [, column_name type [ restrictions ] ...]
  , PRIMARY KEY ( column_name [, column_name ...] ) );
Let's get back to constraints.
... AUTOINCREMENT
This constraint can only be specified as a complement to the PRIMARY KEY, ensuring that identifiers
are incremented all the time. This means that any previous IDs, even those used on deleted entries, will
not be reselected. However, this mechanism is implemented in SQLite less efficiently than a simple
PRIMARY KEY in terms of computing resources and therefore is not recommended for use.
... NOT NULL
This constraint prohibits adding a record to the table in which this field is not filled. By default, when
there is no constraint, any non-unique field can be omitted from the added record and will be set to
NULL.
... CURRENT_TIME
... CURRENT_DATE
... CURRENT_TIMESTAMP
These instructions allow you to automatically populate a field with the time (no date), date (no time),
or full UTC time at the time the record was inserted (provided that the INSERT SQL statement does
not explicitly write anything to this field, even NULL). SQLite does not know how to automatically
detect the time of a record change in a similar way – for this purpose you will have to write a trigger
(which is beyond the scope of the book).
Unfortunately, the CURRENT_TIMESTAMP group restrictions are implemented in SQLite with an
omission: the timestamp is not applied if the field is NULL. This distinguishes SQLite from other SQL
engines and from how SQLite itself handles NULLs in primary key fields. It turns out that for automatic
labeling, you cannot write the entire object to the database, but you need to explicitly specify all the
fields except for the field with the date and time. To solve the problem, we need an alternative option in
which the SQL function STRFTIME('%s') is substituted in the compiled query for the corresponding
columns.
7.6.3 OOP (MQL5) and SQL integration: ORM concept
The use of a database in an MQL program implies that the algorithm is divided into 2 parts: the control
part is written in MQL5, and the execution part is written in SQL. As a result, the source code may

---

## Page 1853

Part 7. Advanced language tools
1 853
7.6 SQLite database
start to look like a patchwork and require attention to maintain consistency. To avoid this, object-
oriented languages have developed the concept of Object-Relational Mapping (ORM), i.e., mapping of
objects to relational table records and vice versa.
The essence of the approach is to encapsulate all actions in the SQL language in classes/structures of
a special layer. As a result, the application part of the program can be written in a pure OOP language
(for example, MQL5), without being distracted by the nuances of SQL.
In the presence of a full-fledged ORM implementation (in the form of a "black box" with a set of all
commands), an application developer generally has the opportunity not to learn SQL.
In addition, ORM allows you to "imperceptibly" change the "engine" of the DBMS if necessary. This is
not particularly relevant for MQL5, because only the SQLite database is built into it, but some
developers prefer to use full-fledged DBMS and connect them to MetaTrader 5 using import of DLLs.
The use of objects with constructors and destructors is very useful when we need to automatically
acquire and release resources. We have covered this concept (RAII, Resource Acquisition Is
Initialization) in the section File descriptor management, however, as we will see later, work with the
database is also based on the allocation and release of different types of descriptors.
The following picture schematically depicts the interaction of different software layers when integrating
OOP and SQL in the form of an ORM.
ORM, Object-Relational Mapping
As a bonus, an object "wrapper" (not just a database-specific ORM) will automate data preparation and
transformation, as well as check for correctness in order to prevent some errors.  
In the following sections, as we walk through the built-in functions for working with the base, we will
implement the examples, gradually building our own simple ORM layer. Due to some specifics of MQL5,

---

## Page 1854

Part 7. Advanced language tools
1 854
7.6 SQLite database
our classes will not be able to provide universalism that covers 1 00% of tasks but will be useful for
many projects.
7.6.4 Creating, opening, and closing databases
The DatabaseOpen and DatabaseClose functions enable the creation and opening of databases.
int DatabaseOpen(const string filename, uint flags)
The function opens or creates a database in a file named filename. The parameter can contain not only
the name but also the path with subfolders relative to MQL5/Files (of a specific terminal instance or in a
shared folder, see flags below). The extension can be omitted, which adds ".sqlite" to the default name.
If NULL or an empty string "" is specified in the filename parameter, then the database is created in a
temporary file, which will be automatically deleted after the database is closed.
If the string ":memory:" is specified in the filename parameter, the database will be created in
memory. Such a temporary base will be automatically deleted after closing.
The flags parameter contains a combination of flags that describe additional conditions for creating or
opening a database from the ENUM_DATABASE_OPEN_FLAGS enumeration.
Identifier
Description
DATABASE_OPEN_READONLY
Open for reading only
DATABASE_OPEN_READWRITE
Open for reading and writing
DATABASE_OPEN_CREATE
Create a file on disk if it doesn't exist
DATABASE_OPEN_MEMORY
Create an in-memory database
DATABASE_OPEN_COMMON
The file is located in the shared folder of all terminals
If none of the DATABASE_OPEN_READONLY or DATABASE_OPEN_READWRITE flags are specified in the
flags parameter, the DATABASE_OPEN_READWRITE flag will be used.
On success, the function returns a handle to the database, which is then used as a parameter for other
functions to access it. Otherwise, INVALID_HANDLE is returned, and the error code can be found in
_ LastError.
void DatabaseClose(int database)
The DatabaseClose function closes the database by its handle, which was previously received from the
DatabaseOpen function.
After calling DatabaseClose, all query handles that we will learn to create for an open base in the
following sections are automatically removed and invalidated.
The function does not return anything. However, if an incorrect handle is passed to it, it will set
_ LastError to ERR_DATABASE_INVALID_HANDLE.
Let's start developing an object-oriented wrapper for databases in a file DBSQLite.mqh.
The DBSQlite class will ensure the creation, opening, and closing of databases. We will extend it later.

---

## Page 1855

Part 7. Advanced language tools
1 855
7.6 SQLite database
class DBSQLite
{
protected:
   const string path;
   const int handle;
   const uint flags;
   
public:
   DBSQLite(const string file, const uint opts =
      DATABASE_OPEN_CREATE | DATABASE_OPEN_READWRITE):
      path(file), flags(opts), handle(DatabaseOpen(file, opts))
   {
   }
   
   ~DBSQLite(void)
   {
      if(handle != INVALID_HANDLE)
      {
         DatabaseClose(handle);
      }
   }
   
   int getHandle() const
   {
      return handle;
   }
   
   bool isOpen() const
   {
      return handle != INVALID_HANDLE;
   }
};
Note that the database is automatically created or opened when the object is created, and closed when
the object is destroyed.
Using this class, let's write a simple script DBinit.mq5, which will create or open the specified database.
input string Database = "MQL5Book/DB/Example1";
   
void OnStart()
{
   DBSQLite db(Database);                   // create or open the base in the constructor
   PRTF(db.getHandle());                    // 65537 / ok
   PRTF(FileIsExist(Database + ".sqlite")); // true / ok
} // the base is closed in the destructor
After the first run, with default settings, we should get a new file
MQL5/Files/MQL5Book/DB/Example1 .sqlite. This is confirmed in the code by checking for the existence
of the file. On subsequent runs with the same name, the script simply opens the database and logs the
current descriptor (an integer number).

---

## Page 1856

Part 7. Advanced language tools
1 856
7.6 SQLite database
7.6.5 Executing queries without MQL5 data binding
Some SQL queries are commands that you just need to send to the engine as is. They require neither
variable input nor results. For example, if our MQL program needs to create a table, index, or view with
a certain structure and name in the database, we can write it as a constant string with the "CREATE
..." statement. In addition, it is convenient to use such queries for batch processing of records or their
combination (merging, calculating aggregated indicators, and same-type modifications). That is, with
one query, you can convert the entire table data or fill other tables based on it. These results can be
analyzed in the subsequent queries.
In all these cases, it is only important to obtain confirmation of the success of the action. Requests of
this type are performed using the DatabaseExecute function.
bool DatabaseExecute(int database, const string sql)
The function executes a query in the database specified by the database descriptor. The request itself
is sent as a ready string sql.
The function returns an indicator of success (true) or error (false).
For example, we can complement our DBSQLite class with this method (the descriptor is already inside
the object).
class DBSQLite
{
   ...
   bool execute(const string sql)
   {
      return DatabaseExecute(handle, sql);
   }
};
Then the script that creates a new table (and, if necessary, beforehand, the database itself) may look
like this (DBcreateTable.mq5).
input string Database = "MQL5Book/DB/Example1";
input string Table = "table1";
   
void OnStart()
{
   DBSQLite db(Database);
   if(db.isOpen())
   {
      PRTF(db.execute(StringFormat("CREATE TABLE %s (msg text)", Table))); // true
   }
}
After executing the script, try to open the specified database in MetaEditor and make sure that it
contains an empty table with a single "msg" text field. But it can also be done programmatically (see
the next section).
If we run the script a second time with the same parameters, we will get an error (albeit a non-critical
one, without forcing the program to close).

---

## Page 1857

Part 7. Advanced language tools
1 857
7.6 SQLite database
database error, table table1 already exists
db.execute(StringFormat(CREATE TABLE %s (msg text),Table))=false / DATABASE_ERROR(5601)
This is because you can't re-create an existing table. But SQL allows you to suppress this error and
create a table only if it hasn't existed yet, otherwise do almost nothing and return a success indicator.
To do this, just add "IF NOT EXISTS" in front of the name in the query.
   db.execute(StringFormat("CREATE TABLE IF NOT EXISTS %s (msg text)", Table));
In practice, tables are required to store information about objects in the application area, such as
quotes, deals, and trading signals. Therefore, it is desirable to automate the creation of tables based on
the description of objects in MQL5. As we will see below, SQLite functions provide the ability to bind
query results to MQL5 structures (but not classes). In this regard, within the framework of the ORM
wrapper, we will develop a mechanism for generating the SQL query "CREATE TABLE" according to the
struct description of the specific type in MQL5.
This requires registering the names and types of structure fields in some way in the general list at the
time of compilation, and then, already at the program execution stage, SQL queries can be generated
from this list.
Several categories of MQL5 entities are parsed at the compilation stage, which can be used to identify
types and names:
• macros
• inheritance
• templates
First of all, it should be recalled that the collected field descriptions are related to the context of a
particular structure and should not be mixed, because the program may contain many different
structures with potentially matching names and types. In other words, it is desirable to accumulate
information in separate lists for each type of structure. A template type is ideal for this, the template
parameter of which (S) will be the application structure. Let's call the template DBEntity.
template<typename S>
struct DBEntity
{
   static string prototype[][3]; // 0 - type, 1 - name, 2 - constraints
   ...
};
   
template<typename T>
static string DBEntity::prototype[][3];
Inside the template, there is a multidimensional array prototype, in which we will write the description
of the fields. To intercept the type and name of the applied field, you will need to declare another
template structure, DBField, inside DBEntity: this time its parameter T is the type of the field itself. In
the constructor, we have information about this type (typename(T)), and we also get the name of the
field (and optionally, the constraint) as parameters.

---

## Page 1858

Part 7. Advanced language tools
1 858
7.6 SQLite database
template<typename S>
struct DBEntity
{
   ...
   template<typename T>
   struct DBField
   {
      T f;
      DBField(const string name, const string constraints = "")
      {
         const int n = EXPAND(prototype);
         prototype[n][0] = typename(T);
         prototype[n][1] = name;
         prototype[n][2] = constraints;
      }
   };
The f field is not used but is needed because structures cannot be empty.
Let's say we have an application structure Data (DBmetaProgramming.mq5).
struct Data
{
   long id;
   string name;
   datetime timestamp;
   double income;
};
We can make its analog inherited from DBEntity<DataDB>, but with substituted fields based on DBField,
identical to the original set.
struct DataDB: public DBEntity<DataDB>
{
   DB_FIELD(long, id);
   DB_FIELD(string, name);
   DB_FIELD(datetime, timestamp);
   DB_FIELD(double, income);
} proto;
By substituting the name of the structure into the parent template parameter, the structure provides
the program with information about its own properties.
Pay attention to the one-time definition of the proto variable along with the structure declaration. This
is necessary because, in templates, each specific parameterized type is compiled only if at least one
object of this type is created in the source code. It is important for us that the creation of this proto-
object occurs at the very beginning of the program launch, at the moment of initialization of global
variables.
A macro is hidden under the DB_FIELD identifier:
#define DB_FIELD(T,N) struct T##_##N: DBField<T> { T##_##N() : DBField<T>(#N) { } } \
   _##T##_##N;
Here's how it expands for a single field:

---

## Page 1859

Part 7. Advanced language tools
1 859
7.6 SQLite database
   struct Type_Name: DBField<Type>
   {
      Type_Name() : DBField<Type>(Name) { }
   } _Type_Name;
Here the structure is not only defined but is also instantly created: in fact, it replaces the original field.
Since the DBField structure contains a single f variable of the desired type, dimensions and internal
binary representation of Data and DataDB are identical. This can be easily verified by running the script
DBmetaProgramming.mq5.
void OnStart()
{
   PRTF(sizeof(Data));
   PRTF(sizeof(DataDB));
   ArrayPrint(DataDB::prototype);
}
It outputs to the log:
DBEntity<Data>::DBField<long>::DBField<long>(const string,const string)
long id
DBEntity<Data>::DBField<string>::DBField<string>(const string,const string)
string name
DBEntity<Data>::DBField<datetime>::DBField<datetime>(const string,const string)
datetime timestamp
DBEntity<Data>::DBField<double>::DBField<double>(const string,const string)
double income
sizeof(Data)=36 / ok
sizeof(DataDB)=36 / ok
            [,0]        [,1]        [,2]
[0,] "long"      "id"        ""         
[1,] "string"    "name"      ""         
[2,] "datetime"  "timestamp" ""         
[3,] "double"    "income"    ""         
However, to access the fields, you would need to write something inconvenient: data._ long_ id.f,
data._ string_ name.f, data._ datetime_ timestamp.f, data._ double_ income.f.
We will not do this, not only and not so much because of inconvenience, but because this way of
constructing meta-structures is not compatible with the principles of data binding to SQL queries. In
the following sections, we will explore database functions that allow you to get records of tables and
results of SQL queries in MQL5 structures. However, it is allowed to use only simple structures without
inheritance and static members of object types. Therefore, it is required to slightly change the principle
of revealing meta-information.
We will have to leave the original types of structures unchanged and actually repeat the description for
the database, making sure that there are no discrepancies (typos). This is not very convenient, but
there is no other way at the moment.
We will transfer the declaration of instances DBEntity and DBField beyond application structures. In this
case, the DB_FIELD macro will receive an additional parameter (S), in which it will be necessary to pass
the type of the application structure (previously it was implicitly taken by declaring it inside the
structure itself).

---

## Page 1860

Part 7. Advanced language tools
1 860
7.6 SQLite database
#define DB_FIELD(S,T,N) \
   struct S##_##T##_##N: DBEntity<S>::DBField<T> \
   { \
      S##_##T##_##N() : DBEntity<S>::DBField<T>(#N) {} \
   }; \
   const S##_##T##_##N _##S##_##T##_##N;
Since table columns can have constraints, they will also need to be passed to the DBField constructor if
necessary. For this purpose, let's add a couple of macros with the appropriate parameters (in theory,
one column can have several restrictions, but usually no more than two).
#define DB_FIELD_C1(S,T,N,C1) \
   struct S##_##T##_##N: DBEntity<S>::DBField<T> \
   {
      S##_##T##_##N() : DBEntity<S>::DBField<T>(#N, C1) {} \
   }; \
   const S##_##T##_##N _##S##_##T##_##N;
   
#define DB_FIELD_C2(S,T,N,C1,C2) \
   struct S##_##T##_##N: DBEntity<S>::DBField<T> \
   { \
      S##_##T##_##N() : DBEntity<S>::DBField<T>(#N, C1 + " " + C2) {} \
   }; \
   const S##_##T##_##N _##S##_##T##_##N;
All three macros, as well as further developments, are added to the header file DBSQLite.mqh.
It is important to note that this "self-made" binding of objects to a table is required only for
entering data into the database because reading data from a table into an object is implemented in
MQL5 using the DatabaseReadBind function.
Let's also improve the implementation of DBField. MQL5 types do not exactly correspond to SQL
storage classes, and therefore it is necessary to perform a conversion when filling the prototype[n][0]
element. This is done by the static method affinity.

---

## Page 1861

Part 7. Advanced language tools
1 861 
7.6 SQLite database
   template<typename T>
   struct DBField
   {
      T f;
      DBField(const string name, const string constraints = "")
      {
         const int n = EXPAND(prototype);
         prototype[n][0] = affinity(typename(T));
         ...
      }
      
      static string affinity(const string type)
      {
         const static string ints[] =
         {
            "bool", "char", "short", "int", "long",
            "uchar", "ushort", "uint", "ulong", "datetime",
            "color", "enum"
         };
         for(int i = 0; i < ArraySize(ints); ++i)
         {
            if(type == ints[i]) return DB_TYPE::INTEGER;
         }
         
         if(type == "float" || type == "double") return DB_TYPE::REAL;
         if(type == "string") return DB_TYPE::TEXT;
         return DB_TYPE::BLOB;
      }
   };
The text constants of SQL generic types used here are placed in a separate namespace: they may be
needed in different places in MQL programs at some point, and it is necessary to ensure that there are
no name conflicts.
namespace DB_TYPE
{
   const string INTEGER = "INTEGER";
   const string REAL = "REAL";
   const string TEXT = "TEXT";
   const string BLOB = "BLOB";
   const string NONE = "NONE";
   const string _NULL = "NULL";
}
Presets of possible restrictions are also described in their group for convenience (as a hint).

---

## Page 1862

Part 7. Advanced language tools
1 862
7.6 SQLite database
namespace DB_CONSTRAINT
{
   const string PRIMARY_KEY = "PRIMARY KEY";
   const string UNIQUE = "UNIQUE";
   const string NOT_NULL = "NOT NULL";
   const string CHECK = "CHECK (%s)"; // requires an expression
   const string CURRENT_TIME = "CURRENT_TIME";
   const string CURRENT_DATE = "CURRENT_DATE";
   const string CURRENT_TIMESTAMP = "CURRENT_TIMESTAMP";
   const string AUTOINCREMENT = "AUTOINCREMENT";
   const string DEFAULT = "DEFAULT (%s)"; // requires an expression (constants, functions)
}
Since some of the constraints require parameters (places for them are marked with the usual '%s'
format modifier), let's add a check for their presence. Here is the final form of the DBField constructor.
   template<typename T>
   struct DBField
   {
      T f;
      DBField(const string name, const string constraints = "")
      {
         const int n = EXPAND(prototype);
         prototype[n][0] = affinity(typename(T));
         prototype[n][1] = name;
         if(StringLen(constraints) > 0       // avoiding error STRING_SMALL_LEN(5035)
            && StringFind(constraints, "%") >= 0)
         {
            Print("Constraint requires an expression (skipped): ", constraints);
         }
         else
         {
            prototype[n][2] = constraints;
         }
      }
Due to the fact that the combination of macros and auxiliary objects DBEntity<S> and DBField<T>
populates an array of prototypes, inside the DBSQlite class, it becomes possible to implement the
automatic generation of an SQL query to create a table of structures.
The createTable method is templated with an application structure type and contains a query stub
("CREATE TABLE %s %s (%s);"). The first argument for it is the optional instruction "IF NOT EXISTS".
The second parameter is the name of the table, which by default is taken as the type of the template
parameter typename(S), but it can be replaced with something else if necessary using the input
parameter name (if it is not NULL). Finally, the third argument in brackets is the list of table columns:
it is formed by the helper method columns based on the array DBEntity <S>::prototype.

---

## Page 1863

Part 7. Advanced language tools
1 863
7.6 SQLite database
class DBSQLite
{
   ...
   template<typename S>
   bool createTable(const string name = NULL,
      const bool not_exist = false, const string table_constraints = "") const
   {
      const static string query = "CREATE TABLE %s %s (%s);";
      const string fields = columns<S>(table_constraints);
      if(fields == NULL)
      {
         Print("Structure '", typename(S), "' with table fields is not initialized");
         SetUserError(4);
         return false;
      }
      // attempt to create an already existing table will give an error,
      // if not using IF NOT EXISTS
      const string sql = StringFormat(query,
         (not_exist ? "IF NOT EXISTS" : ""),
         StringLen(name) ? name : typename(S), fields);
      PRTF(sql);
      return DatabaseExecute(handle, sql);
   }
      
   template<typename S>
   string columns(const string table_constraints = "") const
   {
      static const string continuation = ",\n";
      string result = "";
      const int n = ArrayRange(DBEntity<S>::prototype, 0);
      if(!n) return NULL;
      for(int i = 0; i < n; ++i)
      {
         result += StringFormat("%s%s %s %s",
            i > 0 ? continuation : "",
            DBEntity<S>::prototype[i][1], DBEntity<S>::prototype[i][0],
            DBEntity<S>::prototype[i][2]);
      }
      if(StringLen(table_constraints))
      {
         result += continuation + table_constraints;
      }
      return result;
   }
};
For each column, the description consists of a name, a type, and an optional constraint. Additionally, it
is possible to pass a general constraint on the table (table_ constraints).
Before sending the generated SQL query to the DatabaseExecute function, the createTable method
produces a debug output of the query text to the log (all such output in the ORM classes can be
centrally disabled by replacing the PRTF macro).

---

## Page 1864

Part 7. Advanced language tools
1 864
7.6 SQLite database
Now everything is ready to write a test script DBcreateTableFromStruct.mq5, which, by structure
declaration, would create the corresponding table in SQLite. In the input parameter, we set only the
name of the database, and the program will choose the name of the table itself according to the type of
structure.
#include <MQL5Book/DBSQLite.mqh>
   
input string Database = "MQL5Book/DB/Example1";
   
struct Struct
{
   long id;
   string name;
   double income;
   datetime time;
};
   
DB_FIELD_C1(Struct, long, id, DB_CONSTRAINT::PRIMARY_KEY);
DB_FIELD(Struct, string, name);
DB_FIELD(Struct, double, income);
DB_FIELD(Struct, string, time);
In the main OnStart function, we create a table by calling createTable with default settings. If we do
not want to receive an error sign when we try to create it next time, we need to pass true as the first
parameter (db.createTable<Struct> (true)).
void OnStart()
{
   DBSQLite db(Database);
   if(db.isOpen())
   {
      PRTF(db.createTable<Struct>());
      PRTF(db.hasTable(typename(Struct)));
   }
}
The hasTable method checks for the presence of a table in the database by the table name. We will
consider the implementation of this method in the next section. Now, let's run the script. After the first
run, the table is successfully created and you can see the SQL query in the log (it is displayed with line
breaks, as we formed it in the code).
sql=CREATE TABLE  Struct (id INTEGER PRIMARY KEY,
name TEXT ,
income REAL ,
time TEXT ); / ok
db.createTable<Struct>()=true / ok
db.hasTable(typename(Struct))=true / ok
The second run will return an error from the DatabaseExecute call, because this table already exists,
which is additionally indicated by the hasTable result.

---

## Page 1865

Part 7. Advanced language tools
1 865
7.6 SQLite database
sql=CREATE TABLE  Struct (id INTEGER PRIMARY KEY,
name TEXT ,
income REAL ,
time TEXT ); / ok
database error, table Struct already exists
db.createTable<Struct>()=false / DATABASE_ERROR(5601)
db.hasTable(typename(Struct))=true / ok
7.6.6 Checking if a table exists in the database
The built-in DatabaseTableExists function allows you to check the existence of a table by its name.
bool DatabaseTableExists(int database, const string table)
The database descriptor and the table name are specified in the parameters. The result of the function
call is true if the table exists.
Let's extend the DBSQLite class by adding the hasTable method.
class DBSQLite
{
   ...
   bool hasTable(const string table) const
   {
      return DatabaseTableExists(handle, table);
   }
The script DBcreateTable.mq5 will check if the table has appeared.
void OnStart()
{
   DBSQLite db(Database);
   if(db.isOpen())
   {
      PRTF(db.execute(StringFormat("CREATE TABLE %s (msg text)", Table)));
      PRTF(db.hasTable(Table));
   }
}
Again, don't worry about potentially getting an error when trying to recreate. This does not affect the
existence of the table in any way.
database error, table table1 already exists
db.execute(StringFormat(CREATE TABLE %s (msg text),Table))=false / DATABASE_ERROR(5601)
db.hasTable(Table)=true / ok
Since we are writing a generic helper class DBSQLite, we will provide a mechanism for deleting tables in
it. SQL has the DROP command for this purpose.

---

## Page 1866

Part 7. Advanced language tools
1 866
7.6 SQLite database
class DBSQLite
{
   ...
   bool deleteTable(const string name) const
   {
      const static string query = "DROP TABLE '%s';";
      if(!DatabaseTableExists(handle, name)) return true;
      if(!DatabaseExecute(handle, StringFormat(query, name))) return false;
      return !DatabaseTableExists(handle, name)
         && ResetLastErrorOnCondition(_LastError == DATABASE_NO_MORE_DATA);
   }
   
   static bool ResetLastErrorOnCondition(const bool cond)
   {
      if(cond)
      {
         ResetLastError();
         return true;
      }
      return false;
   }
Before executing the query, we check for the existence of the table and immediately exit if it does not
exist.
After executing the query, we additionally check whether the table has been deleted by calling
DatabaseTableExists again. Since the absence of a table will be flagged with the
DATABASE_NO_MORE_DATA error code, which is the expected result for this method, we clear the
error code with ResetLastErrorOnCondition.
It can be more efficient to use the capabilities of SQL to exclude an attempt to delete a non-existent
table: just add the phrase "IF EXISTS" to the query. Therefore, the final version of the method
deleteTable is simplified:
   bool deleteTable(const string name) const
   {
      const static string query = "DROP TABLE IF EXISTS '%s';";
      return DatabaseExecute(handle, StringFormat(query, name));
   }
You can try to write a test script for deleting the table, but be careful not to delete a working table by
mistake. Tables are deleted immediately with all data, without confirmation and without the possibility
of recovery. For important projects, keep database backups.
7.6.7 Preparing bound queries: DatabasePrepare
In many cases, parameters need to be embedded in SQL queries. Since the SQL query is "originally" a
string that corresponds to a special syntax, it can be formed by a simple StringFormat call or by
concatenation, adding parameter values in the right places. We have already used this technique in
queries to create a table ("CREATE TABLE %s '%s' (%s);"), but here only part of the parameters
contained data (the list of values was substituted for %s inside parentheses), and the rest represented

---

## Page 1867

Part 7. Advanced language tools
1 867
7.6 SQLite database
an option and a table name. In this section, we will focus exclusively on substituting data into a query.
Doing this in a native SQL way is important for several reasons.
First of all, the SQL query is only passed to the SQLite engine as a string, and there it is parsed into
components, checked for correctness, and "compiled" in a certain way (of course, this is not an MQL5
compiler). The compiled query is then executed by the database. That is why we put the word
"originally" in quotation marks.
When the same query needs to be executed with different parameters (for example, inserting many
records into a table; we are slowly approaching this task), separately compiling and checking the query
for each record is rather inefficient. It is more correct to compile the query once, and then execute it
in bulk, simply substituting different values.
This compilation operation is called query preparation and is performed by the DatabasePrepare
function.
Prepared queries have one more purpose: with their help, the SQLite engine returns the results of query
execution to the MQL5 code (you will find more on this in the sections Executing prepared queries and
Separate reading of query result record fields).
The last, but not least, moment associated with parameterized queries is that they protect your
program from potential hacker attacks called SQL injection. First of all, this is critical for databases of
public sites, where information entered by users is recorded in the database by embedding it in SQL
queries: if in this case a simple format substitution '%s' is used, the user will be able to enter some
long string instead of the expected data with additional SQL commands, and it will become part of the
original SQL query, distorting its meaning. But if the SQL query is compiled, it cannot be changed by
the input data: it is always treated as data.
Although the MQL program is not a server program, it can still store information received from the user
in the database.
int DatabasePrepare(int database, const string sql, ...)
The DatabasePrepare function creates a handle in the specified database for the query in the string sql.
The database must be opened beforehand by the DatabaseOpen function.
The query parameter locations are specified in the sql string using fragments '?1 ', '?2', '?3', and so on.
The numbering means the parameter index used in the future when assigning an input value to it, in
DatabaseBind functions. Numbers in the sql string are not required to go in order and can be repeated if
the same parameter needs to be inserted in different places in the query.
Attention! Indexing in substituted fragments '?n' starts from 1 , while in DatabaseBind functions it
starts from 0. For example, the '?1 ' parameter in the query body will get the value when calling
DatabaseBind at index 0, parameter '?2' at index 1 , and so on. This constant offset of 1  is
maintained even if there are gaps (whether it was accidental or intentional) in the numbering of the
'?n' parameters.
If you plan to bind all the parameters strictly in order, you can use an abbreviated notation: in place of
each parameter, simply indicate the symbol '?' without a number: in this case, the parameters are
automatically numbered. Any parameter '?' without a number gets the number which is by 1  larger
than the maximum of the parameters read to the left (with explicit numbers or calculated according to
the same principle, and the very first one will get the number 1 , that is, '?1 ').
Thus, the request 

---

## Page 1868

Part 7. Advanced language tools
1 868
7.6 SQLite database
SELECT * FROM table WHERE risk > ?1 AND signal = ?2
is equivalent to:
SELECT * FROM table WHERE risk > ? AND signal = ?
If some of the parameters are constant or the query is being prepared for one-time execution in order
to get a result, the parameter values can be passed to the DatabasePrepare function as a comma-
separated list instead of an ellipsis (same as in Print or Comment).
Query parameters can only be used to set values in table columns (when writing, changing, or
filtering conditions). Names of tables, columns, options, and SQL keywords cannot be passed
through '?'/'?n' parameters.
The DatabasePrepare function itself does not fulfill the query. The handle returned from it must then be
passed to DatabaseRead or DatabaseReadBind function calls. These functions execute the query and
make the result available for reading (it can be one record or many). Of course, if there are parameter
placeholders ('?' or '?n') in the query, and the values for them were not specified in DatabasePrepare,
before executing the query, you need to bind the parameters and data using the appropriate
DatabaseBind functions.
If a value is not assigned to a parameter, NULL is substituted for it during query execution.
In case of an error, the DatabasePrepare function will return INVALID_HANDLE.
An example of using DatabasePrepare will be introduced in the following sections, after exploring other
features related to prepared queries.
7.6.8 Deleting and resetting prepared queries
Since prepared queries can be executed multiple times, in a loop for different parameter values, it is
required to reset the query to the initial state at each iteration. This is done by the DatabaseReset
function. But it does not make sense to call it if the prepared query is executed once.
bool DatabaseReset(int request)
The function resets the internal compiled query structures to the initial state, similarly to calling
DatabasePrepare. However, DatabaseReset does not recompile the query and is therefore very fast.
It is also important that the function does not reset already established data bindings in the query if
any have been made. Thus, if necessary, you can change the value of only one or a small number of
parameters. Then, after calling DatabaseReset, you can simply call DatabaseBind functions only for
changed parameters.
At the time of writing the book, the MQL5 API did not provide a function to reset the data binding,
an analog of the sqlite_ clear_ bindings function in the standard SQLite distribution.
In the request parameter, specify the valid handle of the query obtained earlier from DatabasePrepare.
If you pass a handle of the query that was previously removed with DatabaseFinalize (see below), an
error will be returned.
The function returns an indicator of success (true) or error (false).
The general principle of working with recurring queries is shown in the following pseudo-code. The
DatabaseBind and DatabaseRead functions will be described in the following sections and will be
"packed" into ORM classes.

---

## Page 1869

Part 7. Advanced language tools
1 869
7.6 SQLite database
struct Data                                       // structure example
{
   long count;
   double value;
   string comment;
};
Data data[];
...                                               // getting data array
int r =
     DatabasePrepare(db, "INSERT... (?, ?, ?)")); // compile query with parameters
for(int i = 0; i < ArraySize(data); ++i)          // data loop
{
   DatabaseBind(r, 0, data[i].count);             // make data binding to parameters
   DatabaseBind(r, 1, data[i].value);
   DatabaseBind(r, 2, data[i].comment);
   DatabaseRead(r);                               // execute request
   ...                                            // analyze or save results
   DatabaseReset(r);                              // initial state at each iteration
}
DatabaseFinalize(r);
After the prepared query is no longer needed, you should release the computer resources it occupies
using DatabaseFinalize.
void DatabaseFinalize(int request)
The function deletes the query with the specified handle, created in DatabasePrepare.
If an incorrect descriptor is passed, the function will record ERR_DATABASE_INVALID_HANDLE to
_ LastError.
When closing the database with DatabaseClose, all query handles created for it are automatically
removed and invalidated.
Let's complement our ORM layer (DBSQLite.mqh) with a new class DBQuery to work with prepared
queries. For now, it will only contain the initialization and deinitialization functionality inherent in the
RAII concept, but we will expand it soon.

---

## Page 1870

Part 7. Advanced language tools
1 870
7.6 SQLite database
class DBQuery
{
protected:
   const string sql;  // query
   const int db;      // database handle (constructor argument)
   const int handle;  // prepared request handle
   
public:
   DBQuery(const int owner, const string s): db(owner), sql(s),
      handle(PRTF(DatabasePrepare(db, sql)))
   {
   }
   
   ~DBQuery()
   {
      DatabaseFinalize(handle);
   }
   
   bool isValid() const
   {
      return handle != INVALID_HANDLE;
   }
   
   virtual bool reset()
   {
      return DatabaseReset(handle);
   }
   ...
};
In the DBSQLite class, we initiate the preparation of the request in the prepare method by creating an
instance of DBQuery. All query objects will be stored in the internal array queries in the form of
autopointers, which allows the calling code not to follow their explicit deletion.
class DBSQLite
{
   ...
protected:
   AutoPtr<DBQuery> queries[];
public:
   DBQuery *prepare(const string sql)
   {
      return PUSH(queries, new DBQuery(handle, sql));
   }
   ...
};
7.6.9 Binding data to query parameters: DatabaseBind/Array
After the SQL query has been compiled by the DatabasePrepare function, you can use the received
query handle to bind data to the query parameters, which is what the DatabaseBind and

---

## Page 1871

Part 7. Advanced language tools
1 871 
7.6 SQLite database
DatabaseBindArray functions are for. Both functions can be called not only immediately after creating a
query in DatabasePrepare but also after resetting the request to its initial state with DatabaseReset (if
the request is executed many times in a loop).
The data binding step is not always required because prepared queries may not have parameters. As a
rule, this situation occurs when a query returns data from SQL to MQL5, and therefore a query
descriptor is required: how to read query results by their handles is described in the sections on
DatabaseRead/DatabaseReadBind and DatabaseColumn-functions.
bool DatabaseBind(int request, int index, T value)
The DatabaseBind function sets the value of the index parameter for the query with the request handle.
By default, numbering starts from 0 if the parameters in the query are marked with substituted
symbols '?' (without a number). However, parameters can be specified in the query string and with a
number (?1 , '?5', ?21 ): in this case, the actual indexes to be passed to the function must be 1  less
than the corresponding number in the string. This is because the numbering in the query string starts
from 1 .
For example, the following query requires one parameter (index 0):
int r = DatabasePrepare(db, "SELECT * FROM table WHERE id=?");
DatabaseBind(r, 0, 1234);
If the "... id=?1 0" substitution were used in the query string, it would be necessary to call
DatabaseBind with index 9.
The value in the DatabaseBind prototype can be of any simple type or string. If a parameter needs to
map composite type data (structures) or arbitrary binary data that can be represented as an array of
bytes, use the DatabaseBindArray function.
The function returns true if successful. Otherwise, it returns false.
bool DatabaseBindArray(int request, int index, T &array[])
The DatabaseBindArray function sets the value of the index parameter as an array of a simple type or
of simple structures (including strings) for the query with the request handle. This function allows you
to write BLOB and NULL (the absence of a value that is considered a separate type in SQL and is not
equal to 0) to the database.
Now let's go back to the DBQuery class in the DBSQLite.mqh file and add data binding support.

---

## Page 1872

Part 7. Advanced language tools
1 872
7.6 SQLite database
class DBQuery
{
   ...
public:
   template<typename T>
   bool bind(const int index, const T value)
   {
      return PRTF(DatabaseBind(handle, index, value));
   }
   template<typename T>
   bool bindBlob(const int index, const T &value[])
   {
      return PRTF(DatabaseBindArray(handle, index, value));
   }
   
   bool bindNull(const int index)
   {
      static const uchar null[] = {};
      return bindBlob(index, null);
   }
   ...
};
BLOB is suitable for transferring any file to the database unchanged, for example, if you first read it into
a byte array using the FileLoad function.
The need to explicitly bind a null value is not so obvious. When inserting new records into the database,
the calling program usually passes only the fields known to it, and all the missing ones (if they are not
marked with the NOT NULL constraint or do not have a different DEFAULT value in the table
description) will be automatically left equal to NULL by the engine. However, when using the ORM
approach, it is convenient to write the entire object to the database, including the field with a unique
primary key (PRIMARY KEY). The new object does not yet have this identifier, since the database itself
adds it when the object is first written, so it is important to bind this field in the new object to the NULL
value.
7.6.1 0 Executing prepared queries: DatabaseRead/Bind
Prepared queries are executed using the DatabaseRead and DatabaseReadBind functions. The first
function extracts the results from the database in such a way that later individual fields can be read
from each record received in turn in response, and the second extracts each matching record in its
entirety, in the form of a structure.
bool DatabaseRead(int request)
On the first call, after Database Prepare or DatabaseReset, the DatabaseRead function executes the
query and sets the internal query result pointer to the first record retrieved (if the query expects
records to be returned). The DatabaseColumn functions enable the reading of the values of the record
fields, i.e., the columns specified in the query.
On subsequent calls, the DatabaseRead function jumps to the next record in the query results until the
end is reached.

---

## Page 1873

Part 7. Advanced language tools
1 873
7.6 SQLite database
The function returns true upon successful completion. The false value is used as an indicator of an error
(for example, the database may be blocked or busy), as well as when the end of the results is normally
reached, so you should analyze the code in _ LastError. In particular, the value
ERR_DATABASE_NO_MORE_DATA (51 26) indicates that the results are finished.
Attention! If DatabaseRead is used to execute queries that don't return data, such as INSERT,
UPDATE, etc., the function immediately returns false and sets the error code
ERR_DATABASE_NO_MORE_DATA if the request was successful.
The usual pattern of using the function is illustrated by the following pseudo-code (DatabaseColumn
functions for different types are presented in the next section).
int r = DatabasePrepare(db, "SELECT... WHERE...?",
   param));                            //compiling the query(optional with parameters)
while(DatabaseRead(r))                 // query execution (on the first iteration)
{                                      //    and loop through result records
   int count;
   DatabaseColumnInteger(r, 0, count); // read one field from the current record
   double number;
   DatabaseColumnDouble(r, 1, number); // read another field from the current record
   ...                                 // column types and numbers in record are determined by program
                                       // process the received values of count, number, etc.
}                                      // loop is interrupted when the end of the results is reached
DatabaseFinalize(r);
Note that since the query (reading conditional data) is actually executed only once (on the very first
iteration), there is no need to call DatabaseReset, as we did when recording changing data. However, if
we want to run the query again and "walk" through the new results, calling DatabaseReset would be
necessary.
bool DatabaseReadBind(int request, void &object)
The DatabaseReadBind function works in a similar way to DatabaseRead: the first call executes the SQL
query and, in case of success (there is suitable data in the result), fills the obj ect structure passed by
reference with fields of the first record; subsequent calls continue moving the internal pointer through
the records in the query results, filling the structure with the data of the next record.
The structure must have only numeric types and/or strings as members (arrays are not allowed), it
cannot cannot inherit from or contain static members of object types.
The number of fields in the obj ect structure should not exceed the number of columns in the query
results; otherwise, we will get an error. The number of columns can be found dynamically using the
DatabaseColumnsCount function, however, the caller usually needs to "know" in advance the expected
data configuration according to the original request.
If the number of fields in the structure is less than the number of fields in the record, a partial read will
be performed. The rest of the data can be obtained using the appropriate DatabaseColumn functions.
It is assumed that the field types of the structure match the data types in the result columns.
Otherwise, an automatic implicit conversion will be performed, which can lead to unexpected
consequences (for example, a string read into a numeric field will give 0).
In the simplest case, when we calculate a certain total value for the database records, for example, by
calling an aggregate function like SUM(column), COUNT(column), or AVERAGE(column), the result of
the query will be a single record with a single field.

---

## Page 1874

Part 7. Advanced language tools
1 874
7.6 SQLite database
SELECT SUM(swap) FROM trades;
Because reading the results is related to DatabaseColumn functions, we will defer the development of
the example until the next section, where they are presented.
7.6.1 1  Reading fields separately: DatabaseColumn Functions
As a result of query execution by the DatabaseRead or DatabaseReadBind functions, the program gets
the opportunity to scroll through the records selected according to the specified conditions. At each
iteration, in the internal structures of the SQLite engine, one specific record is allocated, the fields
(columns) of which are available through the group of DatabaseColumn functions.
int DatabaseColumnsCount(int request)
Based on the query descriptor, the function returns the number of fields (columns) in the query results.
In case of an error, it returns -1 .
You can find out the number of fields in the query created in DatabasePrepare even before calling the
DatabaseRead function. For other DatabaseColumn functions, you should initially call DatabaseRead (at
least once). 
Using the original number of a field in the query results, the program can find the field name
(DatabaseColumnName), type (DatabaseColumnType), size (DatabaseColumnSize), and the value of the
corresponding type (each type has its function).
bool DatabaseColumnName(int request, int column, string &name)
The function fills the string parameter passed by reference (name) with the name of the column
specified by number (column) in the query results (request).
Field numbering starts from 0 and cannot exceed the value of DatabaseColumnsCount() - 1 . This applies
not only to this function but also to all other functions of the section.
The function returns true if successful or false in case of an error.
ENUM_DATABASE_FIELD_TYPE DatabaseColumnType(int request, int column)
The DatabaseColumnType function returns the type of the value in the specified column in the current
record of the query results. The possible types are collected in the ENUM_DATABASE_FIELD_TYPE
enumeration.
Identifier
Description
DATABASE_FIELD_TYPE_INVALID
Error getting type, error code in _LastError
DATABASE_FIELD_TYPE_INTEGER
Integer number
DATABASE_FIELD_TYPE_FLOAT
Real number
DATABASE_FIELD_TYPE_TEXT
String
DATABASE_FIELD_TYPE_BLOB
Binary data
DATABASE_FIELD_TYPE_NULL
Void (special type NULL)

---

## Page 1875

Part 7. Advanced language tools
1 875
7.6 SQLite database
More details about SQL types and their correspondence to MQL5 types were described in the section
Structure (schema) of tables: data types and restrictions.
int DatabaseColumnSize(int request, int column)
The function returns the size of the value in bytes for the field with the column index in the current
record of results of the request query. For example, integer values can be represented by a different
number of bytes (we know this from MQL5 types, in particular, short/int/long).
The next group of functions allows you to get the value of a particular type from the corresponding field
of the record. To read values from the next record, you need to call DatabaseRead again.
bool DatabaseColumnText(int request, int column, string &value)
bool DatabaseColumnInteger(int request, int column, int &value)
bool DatabaseColumnLong(int request, int column, long &value)
bool DatabaseColumnDouble(int request, int column, double &value)
bool DatabaseColumnBlob(int request, int column, void &data[])
All functions return true on success and put the field value in the receiving variable value. The only
special case is the function DatabaseColumnBlob, which passes an array of an arbitrary simple type or
simple structures as an output variable. By specifying the uchar[] array as the most versatile option,
you can read the byte representation of any value (including binary files marked with the
DATABASE_FIELD_TYPE_BLOB type).
The SQLite engine does not check that for a column a function corresponding to its type is called.
If the types are inadvertently or intentionally different, the system will automatically implicitly
convert the field value to the type of the receiving variable.
Now, after getting familiar with the majority of Database functions, we can complete the development
of a set of SQL classes in the DBSQLite.mqh file and proceed to practical examples.
7.6.1 2 Examples of CRUD operations in SQLite via ORM objects
We have studied all the functions required for the implementation of the complete lifecycle of
information in the database, that is CRUD (Create, Read, Update, Delete). But before proceeding to
practice, we need to complete the ORM layer.
From the previous few sections, it is already clear that the unit of work with the database is a record: it
can be a record in a database table or an element in the results of a query. To read a single record at
the ORM level, let's introduce the DBRow class. Each record is generated by an SQL query, so its
handle is passed to the constructor.
As we know, a record can consist of several columns, the number and types of which allow us to find
DatabaseColumn functions. To expose this information to an MQL program using DBRow, we reserved
the relevant variables: columns and an array of structures DBRowColumn (the last one contains three
fields for storing the name, type, and size of the column).
In addition, DBRow objects may, if necessary, cache in themselves the values obtained from the
database. For this purpose, the data array of type MqlParam is used. Since we do not know in advance
what type of values will be in a particular column, we use MqlParam as a kind of universal type Variant
available in other programming environments.

---

## Page 1876

Part 7. Advanced language tools
1 876
7.6 SQLite database
class DBRow
{
protected:
   const int query; 
   int columns;
   DBRowColumn info[];
   MqlParam data[];
   const bool cache;
   int cursor;
   ...
public:
   DBRow(const int q, const bool c = false):
      query(q), cache(c), columns(0), cursor(-1)
   {
   }
   
   int length() const
   {
      return columns;
   }
   ...
};
The cursor variable tracks the current record number from the query results. Until the request is
completed, cursor equals -1 .
The virtual method DBread is responsible for executing the query; it calls DatabaseRead.
protected:
   virtual bool DBread()
   {
      return PRTF(DatabaseRead(query));
   }
We will see later why we needed a virtual method. The public method next, which uses DBread, provides
"scrolling" through the result records and looks like this.

---

## Page 1877

Part 7. Advanced language tools
1 877
7.6 SQLite database
public:
   virtual bool next()
   {
      ...
      const bool success = DBread();
      if(success)
      {
         if(cursor == -1)
         {
            columns = DatabaseColumnsCount(query);
            ArrayResize(info, columns);
            if(cache) ArrayResize(data, columns);
            for(int i = 0; i < columns; ++i)
            {
               DatabaseColumnName(query, i, info[i].name);
               info[i].type = DatabaseColumnType(query, i);
               info[i].size = DatabaseColumnSize(query, i);
               if(cache) data[i] = this[i]; // overload operator[](int)
            }
         }
         ++cursor;
      }
      return success;
   }
If the query is accessed for the first time, we allocate memory and read the column information. If
caching was requested, we additionally populate the data array. To do this, the overloaded operator '[]'
is called for each column. In it, depending on the type of value, we call the appropriate
DatabaseColumn function and put the resulting value in one or another field of the MqlParam structure.

---

## Page 1878

Part 7. Advanced language tools
1 878
7.6 SQLite database
   virtual MqlParam operator[](const int i = 0) const
   {
      MqlParam param = {};
      if(i < 0 || i >= columns) return param;
      if(ArraySize(data) > 0 && cursor != -1) // if there is a cache, return from it
      {
         return data[i];
      }
      switch(info[i].type)
      {
      case DATABASE_FIELD_TYPE_INTEGER:
         switch(info[i].size)
         {
         case 1:
            param.type = TYPE_CHAR;
            break;
         case 2:
            param.type = TYPE_SHORT;
            break;
         case 4:
            param.type = TYPE_INT;
            break;
         case 8:
         default:
            param.type = TYPE_LONG;
            break;
         }
         DatabaseColumnLong(query, i, param.integer_value);
         break;
      case DATABASE_FIELD_TYPE_FLOAT:
         param.type = info[i].size == 4 ? TYPE_FLOAT : TYPE_DOUBLE;
         DatabaseColumnDouble(query, i, param.double_value);
         break;
      case DATABASE_FIELD_TYPE_TEXT:
         param.type = TYPE_STRING;
         DatabaseColumnText(query, i, param.string_value);
         break;
      case DATABASE_FIELD_TYPE_BLOB: // return base64 only for information we can't
         {                           // return binary data in MqlParam - exact 
            uchar blob[];            // representation of binary fields is given by getBlob 
            DatabaseColumnBlob(query, i, blob);
            uchar key[], text[];
            if(CryptEncode(CRYPT_BASE64, blob, key, text))
            {
               param.string_value = CharArrayToString(text);
            }
         }
         param.type = TYPE_BLOB;
         break;
      case DATABASE_FIELD_TYPE_NULL:
         param.type = TYPE_NULL;

---

## Page 1879

Part 7. Advanced language tools
1 879
7.6 SQLite database
         break;
      }
      return param;
   }
The getBlob method is provided to fully read binary data from BLOB fields (use type uchar as S to get a
byte array if there is no more specific information about the content format).
   template<typename S>
   int getBlob(const int i, S &object[])
   {
      ...
      return DatabaseColumnBlob(query, i, object);
   }
For the described methods, the process of executing a query and reading its results can be represented
by the following pseudo-code (it leaves behind the scenes the existing DBSQLite and DBQuery classes,
but we will bring them all together soon):
int query = ...
DBRow *row = new DBRow(query);
while(row.next())
{
   for(int i = 0; i < row.length(); ++i)
   {
      StructPrint(row[i]); // print the i-th column as an MqlParam structure
   }
}
It is not elegant to explicitly write a loop through the columns every time, so the class provides a
method for obtaining the values of all fields of the record.
   void readAll(MqlParam &params[]) const
   {
      ArrayResize(params, columns);
      for(int i = 0; i < columns; ++i)
      {
         params[i] = this[i];
      }
   }
Also, the class received for convenience overloads of the operator '[]' and the getBlob method for
reading fields by their names instead of indexes. For example,

---

## Page 1880

Part 7. Advanced language tools
1 880
7.6 SQLite database
class DBRow
{
   ...
public:
   int name2index(const string name) const
   {
      for(int i = 0; i < columns; ++i)
      {
         if(name == info[i].name) return i;
      }
      Print("Wrong column name: ", name);
      SetUserError(3);
      return -1;
   }
   
   MqlParam operator[](const string name) const
   {
      const int i = name2index(name);
      if(i != -1) return this[i]; // operator()[int] overload
      static MqlParam param = {};
      return param;
   }
   ...
};
This way you can access selected columns.
int query = ...
DBRow *row = new DBRow(query);
for(int i = 1; row.next(); )
{
   Print(i++, " ", row["trades"], " ", row["profit"], " ", row["drawdown"]);
}
But still getting the elements of the record individually, as a MqlParam array, can not be called a truly
OOP approach. It would be preferable to read the entire database table record into an object, an
application structure. Recall that the MQL5 API provides a suitable function: DatabaseReadBind. This is
where we get the advantage of the ability to describe a derived class DBRow and override its virtual
method DBRead.
This class of DBRowStruct is a template and expects as parameter S one of the simple structures
allowed to be bound in DatabaseReadBind.

---

## Page 1881

Part 7. Advanced language tools
1 881 
7.6 SQLite database
template<typename S>
class DBRowStruct: public DBRow
{
protected:
   S object;
   
   virtual bool DBread() override
   {
      // NB: inherited structures and nested structures are not allowed;
      // count of structure fields should not exceed count of columns in table/query
      return PRTF(DatabaseReadBind(query, object));
   }
   
public:
   DBRowStruct(const int q, const bool c = false): DBRow(q, c)
   {
   }
   
   S get() const
   {
      return object;
   }
};
With a derived class, we can get objects from the base almost seamlessly.
int query = ...
DBRowStruct<MyStruct> *row = new DBRowStruct<MyStruct>(query);
MyStruct structs[];
while(row.next())
{
   PUSH(structs, row.get());
}
Now it's time to turn the pseudo-code into working code by linking DBRow/DBRowStruct with DBQuery.
In DBQuery, we add an autopointer to the DBRow object, which will contain data about the current
record from the results of the query (if it was executed). Using an autopointer frees the calling code
from worrying about freeing DBRow objects: they are deleted either with DBQuery or when re-created
due to query restart (if required). The initialization of the DBRow or DBRowStruct object is completed
by a template method start.

---

## Page 1882

Part 7. Advanced language tools
1 882
7.6 SQLite database
class DBQuery
{
protected:
   ...
   AutoPtr<DBRow> row;    // current entry
public:
   DBQuery(const int owner, const string s): db(owner), sql(s),
      handle(PRTF(DatabasePrepare(db, sql)))
   {
      row = NULL;
   }
   
   template<typename S>
   DBRow *start()
   {
      DatabaseReset(handle);
      row = typename(S) == "DBValue" ? new DBRow(handle) : new DBRowStruct<S>(handle);
      return row[];
   }
The DBValue type is a dummy structure that is needed only to instruct the program to create the
underlying DBRow object, without violating the compilability of the line with the DatabaseReadBind call.
With the start method, all of the above pseudo-code fragments become working due to the following
preparation of the request:
DBSQLite db("MQL5Book/DB/Example1");                            // open base
DBQuery *query = db.prepare("PRAGMA table_xinfo('Struct')");    // prepare the request
DBRowStruct<DBTableColumn> *row = query.start<DBTableColumn>(); // get object cursor 
DBTableColumn columns[];                                        // receiving array of objects
while(row.next())             // loop while there are records in the query result
{
   PUSH(columns, row.get());  // getting an object from the current record
}
ArrayPrint(columns);
This example reads meta-information about the configuration of a particular table from the database
(we created it in the example DBcreateTableFromStruct.mq5 in the section Executing queries without
MQL5 data binding): each column is described by a separate record with several fields (SQLite
standard), which is formalized in the structure DBTableColumn.
struct DBTableColumn
{
   int cid;              // identifier (serial number)
   string name;          // name
   string type;          // type
   bool not_null;        // attribute NOT NULL (yes/no)
   string default_value; // default value
   bool primary_key;     // PRIMARY KEY sign (yes/no)
};
To save the user from having to write a loop every time with the translation of results records into
structure objects, the DBQuery class provides a template method readAll that populates a referenced

---

## Page 1883

Part 7. Advanced language tools
1 883
7.6 SQLite database
array of structures with information from the query results. A similar readAll method fills an array of
pointers to DBRow objects (this is more suitable for receiving the results of synthetic queries with
columns from different tables).
In a quartet of operations, the CRUD method DBRowStruct::get is responsible for the letter R (Read). To
make the reading of an object more functionally complete, we will support point recovery of an object
from the database by its identifier.
The vast majority of tables in SQLite databases have a primary key rowid (unless the developer for one
reason or another used the "WITHOUT ROWID" option in the description), so the new read method will
take a key value as a parameter. By default, the name of the table is assumed to be equal to the type
of the receiving structure but can be changed to an alternative one through the table parameter.
Considering that such a request is a one-time request and should return one record, it makes sense to
place the read method directly to the class DBSQLite and manage short-lived objects DBQuery and
DBRowStruct<S> inside.
class DBSQLite
{
   ...
public:
   template<typename S>
   bool read(const long rowid, S &s, const string table = NULL,
      const string column = "rowid")
   {
      const static string query = "SELECT * FROM '%s' WHERE %s=%ld;";
      const string sql = StringFormat(query,
         StringLen(table) ? table : typename(S), column, rowid);
      PRTF(sql);
      DBQuery q(handle, sql);
      if(!q.isValid()) return false;
      DBRowStruct<S> *r = q.start<S>();
      if(r.next())
      {
         s = r.get();
         return true;
      }
      return false;
   }
};
The main work is done by the SQL query "SELECT * FROM '%s' WHERE %s=%ld;", which returns a
record with all fields from the specified table by matching the rowid key.
Now you can create a specific object from the database like this (it is assumed that the identifier of
interest to us must be stored somewhere).

---

## Page 1884

Part 7. Advanced language tools
1 884
7.6 SQLite database
   DBSQLite db("MQL5Book/DB/Example1");
   long rowid = ... // ill in the identifier
   Struct s; 
   if(db.read(rowid, s))
      StructPrint(s);
Finally, in some complex cases where maximum flexibility in querying is required (for example, a
combination of several tables, usually a SELECT with a JOIN, or nested queries), we still have to allow
an explicit SQL command to get a selection, although this violates the ORM principle. This possibility is
opened by the method DBSQLite::prepare, which we have already presented in the context of the
management of prepared queries.
We have considered all the main ways of reading.
However, we don't have anything to read from the database yet, because we skipped the step of adding
records.
Let's try to implement object creation (C). Recall that in our object concept, structure types semi-
automatically define database tables (using DB_FIELD macros). For example, the Struct structure
allowed the creation of a "Struct" table in the database with a set of columns corresponding to the
fields of the structure. We provided this with a template method createTable in the DBSQLite class.
Now, by analogy, you need to write a template method insert, which would add a record to this table.
An object of a structure is passed to the method, for the type of which the filled
DBEntity<S>::prototype <S> array must exist (it is filled with macros). Thanks to this array, we can
form a list of parameters (more precisely, their substitutes '?n'): this is done by the static method qlist.
However, the preparation of the query is still half a battle. In the code below, we will need to bind the
input data based on the properties of the object.
A "RETURNING rowid" statement has been added to the "INSERT" command, so when the query
succeeds, we expect a single result row with one value: new rowid.

---

## Page 1885

Part 7. Advanced language tools
1 885
7.6 SQLite database
class DBSQLite
{
   ...
public:
   template<typename S>
   long insert(S &object, const string table = NULL)
   {
      const static string query = "INSERT INTO '%s' VALUES(%s) RETURNING rowid;";
      const int n = ArrayRange(DBEntity<S>::prototype, 0);
      const string sql = StringFormat(query,
         StringLen(table) ? table : typename(S), qlist(n));
      PRTF(sql);
      DBQuery q(handle, sql);
      if(!q.isValid()) return 0;
      DBRow *r = q.start<DBValue>();
      if(object.bindAll(q))
      {
         if(r.next()) // the result should be one record with one new rowid value
         {
            return object.rowid(r[0].integer_value);
         }
      }
      return 0;
   }
   
   static string qlist(const int n)
   {
      string result = "?1";
      for(int i = 1; i < n; ++i)
      {
         result += StringFormat(",?%d", (i + 1));
      }
      return result;
   }
};
The source code of the insert method has one point to which special attention should be paid. To bind
values to query parameters, we call the obj ect.bindAll(q) method. This means that in the application
structure that you want to integrate with the base, you need to implement such a method that provides
all member variables for the engine.
In addition, to identify objects, it is assumed that there is a field with a primary key, and only the
object "knows" what this field is. So, the structure has the rowid method, which serves a dual action:
first, it transfers the record identifier assigned in the database to the object, and second, it allows
finding out this identifier from the object, if it has already been assigned earlier.
The DBSQLite::update (U) method for changing a record is similar in many ways to insert, and therefore
it is proposed to familiarize yourself with it. Its basis is the SQL query "UPDATE '%s' SET (%s)=(%s)
WHERE rowid=%ld;", which is supposed to pass all the fields of the structure (bindAll() object) and key
(rowid() object).

---

## Page 1886

Part 7. Advanced language tools
1 886
7.6 SQLite database
Finally, we mention that the point deletion (D) of a record by an object is implemented in the method
DBSQLite::remove (word delete is an MQL5 operator).
Let's show all methods in an example script DBfillTableFromStructArray.mq5, where the Struct new
structure is defined.
We will make several values of commonly used types as fields of the structure.
struct Struct
{
   long id;
   string name;
   double number;
   datetime timestamp;
   string image;
   ...
};
In the string field image, the calling code will specify the name of the graphic resource or the name of
the file, and at the time of binding to the database, the corresponding binary data will be copied as a
BLOB. Subsequently, when we read data from the database into Struct objects, the binary data will end
up in the image string but, of course, with distortions (because the line will break on the first null byte).
To accurately extract BLOBs from the database, you will need to call the method DBRow::getBlob
(based on DatabaseColumnBlob).
Creating meta-information about fields of the Struct structure provides the following macros. Based on
them, an MQL program can automatically create a table in the database for Struct objects, as well as
initiate the binding of the data passed to the queries based on the properties of the objects (this binding
should not be confused with the reverse binding for obtaining query results, i.e. DatabaseReadBind).
DB_FIELD_C1(Struct, long, id, DB_CONSTRAINT::PRIMARY_KEY);
DB_FIELD(Struct, string, name);
DB_FIELD(Struct, double, number);
DB_FIELD_C1(Struct, datetime, timestamp, DB_CONSTRAINT::CURRENT_TIMESTAMP);
DB_FIELD(Struct, blob, image);
To fill a small test array of structures, the script has input variables: they specify a trio of currencies
whose quotes will fall into the number field. We have also embedded two standard images into the script
in order to test the work with BLOBs: they will "go" to the image field. The timestamp field will be
automatically populated by our ORM classes with the current insertion or modification timestamp of the
record. The primary key in the id field will have to be populated by SQLite itself.
#resource "\\Images\\euro.bmp"
#resource "\\Images\\dollar.bmp"
   
input string Database = "MQL5Book/DB/Example2";
input string EURUSD = "EURUSD";
input string USDCNH = "USDCNH";
input string USDJPY = "USDJPY";
Since the values for the input query variables (those same '?n') are bound, ultimately, using the
functions DatabaseBind or DatabaseBindArray under the numbers, our bindAll structure in the method
should establish a correspondence between the numbers and their fields: a simple numbering is
assumed in the order of declaration.

---

## Page 1887

Part 7. Advanced language tools
1 887
7.6 SQLite database
struct Struct
{
   ...
   bool bindAll(DBQuery &q) const
   {
      uint pixels[] = {};
      uint w, h;
      if(StringLen(image))                // load binary data
      {
         if(StringFind(image, "::") == 0) // this is a resource
         {
            ResourceReadImage(image, pixels, w, h);
            // debug/test example (not BMP, no header)
            FileSave(StringSubstr(image, 2) + ".raw", pixels);
         }
         else                             // it's a file
         {
            const string res = "::" + image;
            ResourceCreate(res, image);
            ResourceReadImage(res, pixels, w, h);
            ResourceFree(res);
         }
      }
      // when id = NULL, the base will assign a new rowid
      return (id == 0 ? q.bindNull(0) : q.bind(0, id))
         && q.bind(1, name)
         && q.bind(2, number)
         // && q.bind(3, timestamp) // this field will be autofilled CURRENT_TIMESTAMP
         && q.bindBlob(4, pixels);
   }
   ...
};
Method rowid is very simple.
struct Struct
{
   ...
   long rowid(const long setter = 0)
   {
      if(setter) id = setter;
      return id;
   }
};
Having defined the structure, we describe a test array of 4 elements. Only 2 of them have attached
images. All objects have zero identifiers because they are not yet in the database.

---

## Page 1888

Part 7. Advanced language tools
1 888
7.6 SQLite database
Struct demo[] =
{
   {0, "dollar", 1.0, 0, "::Images\\dollar.bmp"},
   {0, "euro", SymbolInfoDouble(EURUSD, SYMBOL_ASK), 0, "::Images\\euro.bmp"},
   {0, "yuan", 1.0 / SymbolInfoDouble(USDCNH, SYMBOL_BID), 0, NULL},
   {0, "yen", 1.0 / SymbolInfoDouble(USDJPY, SYMBOL_BID), 0, NULL},
};
In the main OnStart function, we create or open a database (by default MQL5Book/DB/Example2.sqlite).
Just in case, we try to delete the "Struct" table in order to ensure reproducibility of the results and
debugging when the script is repeated, then we will create a table for the Struct structure.
void OnStart()
{
   DBSQLite db(Database);
   if(!PRTF(db.isOpen())) return;
   PRTF(db.deleteTable(typename(Struct)));
   if(!PRTF(db.createTable<Struct>(true))) return;
   ...
Instead of adding objects one at a time, we use a loop:
 // -> this option (set aside)
   for(int i = 0; i < ArraySize(demo); ++i)
   {
      PRTF(db.insert(demo[i])); // get a new rowid on each call
   }
In this loop, we will use an alternative implementation of the insert method, which takes an array of
objects as input at once and processes them in a single request, which is more efficient (but the
general ditch of the method is the previously considered insert method for one object).
   db.insert(demo);  // new rowids are placed in objects
   ArrayPrint(demo);
   ...
Now let's try to select records from the database according to some conditions, for example, those
that do not have an image assigned. To do this, let's prepare an SQL query wrapped in the DBQuery
object, and then we get its results in two ways: through binding to Struct structures or via the
instances of the generic class DBRow.

---

## Page 1889

Part 7. Advanced language tools
1 889
7.6 SQLite database
   DBQuery *query = db.prepare(StringFormat("SELECT * FROM %s WHERE image IS NULL",
      typename(Struct)));
   
   // approach 1: application type of the Struct structure
   Struct result[];
   PRTF(query.readAll(result));
   ArrayPrint(result);
   
   query.reset(); // reset the query to try again
   
   // approach 2: generic DBRow record container with MqlParam values
   DBRow *rows[];
   query.readAll(rows); // get DBRow objects with cached values
   for(int i = 0; i < ArraySize(rows); ++i)
   {
      Print(i);
      MqlParam fields[];
      rows[i].readAll(fields);
      ArrayPrint(fields);
   }
   ...
Both options should give the same result, albeit presented differently (see the log below).
Next, our script pauses for 1  second so that we can notice the changes in the timestamps of the next
entries that we will change.
   Print("Pause...");
   Sleep(1000);
   ...
To objects in the result[] array, we assign the "yuan.bmp" image located in the folder next to the
script. Then we update the objects in the database.
   for(int i = 0; i < ArraySize(result); ++i)
   {
      result[i].image = "yuan.bmp";
      db.update(result[i]);
   }
   ...
After running the script, you can make sure that all four records have BLOBs in the database navigator
built into MetaEditor, as well as the difference in timestamps for the first two and the last two records.
Let's demonstrate the extraction of binary data. We will first see how a BLOB is mapped to the image
string field (binary data is not for the log, we only do this for demonstration purposes).

---

## Page 1890

Part 7. Advanced language tools
1 890
7.6 SQLite database
   const long id1 = 1;
   Struct s;
   if(db.read(id1, s))
   {
      Print("Length of string with Blob: ", StringLen(s.image));
      Print(s.image);
   }
   ...
Then we read the entire data with getBlob (total length is greater than the line above).
   DBRow *r;
   if(db.read(id1, r, "Struct"))
   {
      uchar bytes[];
      Print("Actual size of Blob: ", r.getBlob("image", bytes));
      FileSave("temp.bmp.raw", bytes); // not BMP, no header
   }
We need to get the temp.bmp.raw file, identical to MQL5/Files/Images/dollar.bmp.raw, which is created
in the method Struct::bindAll for debugging purposes. Thus, it is easy to verify the exact
correspondence of written and read binary data.
Note that since we are storing the resource's binary content in the database, it is not a BMP source
file: resources produce color normalization and store a headerless array of pixels with meta-
information about the image.
While running, the script generates a detailed log. In particular, the creation of a database and a table
is marked with the following lines.
db.isOpen()=true / ok
db.deleteTable(typename(Struct))=true / ok
sql=CREATE TABLE IF NOT EXISTS Struct (id INTEGER PRIMARY KEY,
name TEXT ,
number REAL ,
timestamp INTEGER CURRENT_TIMESTAMP,
image BLOB ); / ok
db.createTable<Struct>(true)=true / ok
The SQL query for inserting an array of objects is prepared once and then executed many times with
pre-binding different data (only one iteration is shown here). The number of DatabaseBind function calls
matches the '?n' variables in the query ('?4' is automatically replaced by our classes with the SQL
STRFTIME('%s') function call to get the current UTC timestamp).
sql=INSERT INTO 'Struct' VALUES(?1,?2,?3,STRFTIME('%s'),?5) RETURNING rowid; / ok
DatabasePrepare(db,sql)=131073 / ok
DatabaseBindArray(handle,index,value)=true / ok
DatabaseBind(handle,index,value)=true / ok
DatabaseBind(handle,index,value)=true / ok
DatabaseBindArray(handle,index,value)=true / ok
DatabaseRead(query)=true / ok
...
Next, an array of structures with already assigned primary keys rowid is output to the log in the first
column.

---

## Page 1891

Part 7. Advanced language tools
1 891 
7.6 SQLite database
    [id]   [name] [number]         [timestamp]               [image]
[0]    1 "dollar"  1.00000 1970.01.01 00:00:00 "::Images\dollar.bmp"
[1]    2 "euro"    1.00402 1970.01.01 00:00:00 "::Images\euro.bmp"  
[2]    3 "yuan"    0.14635 1970.01.01 00:00:00 null                 
[3]    4 "yen"     0.00731 1970.01.01 00:00:00 null
Selecting records without images gives the following result (we execute this query twice with different
methods: the first time we fill the array of Struct structures, and the second is the DBRow array, from
which for each field we get the "value" in the form of MqlParam).
DatabasePrepare(db,sql)=196609 / ok
DatabaseReadBind(query,object)=true / ok
DatabaseReadBind(query,object)=true / ok
DatabaseReadBind(query,object)=false / DATABASE_NO_MORE_DATA(5126)
query.readAll(result)=true / ok
    [id] [name] [number]         [timestamp] [image]
[0]    3 "yuan"  0.14635 2022.08.20 13:14:38 null   
[1]    4 "yen"   0.00731 2022.08.20 13:14:38 null   
DatabaseRead(query)=true / ok
DatabaseRead(query)=true / ok
DatabaseRead(query)=false / DATABASE_NO_MORE_DATA(5126)
0
    [type] [integer_value] [double_value] [string_value]
[0]      4               3        0.00000 null          
[1]     14               0        0.00000 "yuan"        
[2]     13               0        0.14635 null          
[3]     10      1661001278        0.00000 null          
[4]      0               0        0.00000 null          
1
    [type] [integer_value] [double_value] [string_value]
[0]      4               4        0.00000 null          
[1]     14               0        0.00000 "yen"         
[2]     13               0        0.00731 null          
[3]     10      1661001278        0.00000 null          
[4]      0               0        0.00000 null          
...
The second part of the script updates a couple of found records without images and adds BLOBs to
them.

---

## Page 1892

Part 7. Advanced language tools
1 892
7.6 SQLite database
Pause...
sql=UPDATE 'Struct' SET (id,name,number,timestamp,image)=
   (?1,?2,?3,STRFTIME('%s'),?5) WHERE rowid=3; / ok
DatabasePrepare(db,sql)=262145 / ok
DatabaseBind(handle,index,value)=true / ok
DatabaseBind(handle,index,value)=true / ok
DatabaseBind(handle,index,value)=true / ok
DatabaseBindArray(handle,index,value)=true / ok
DatabaseRead(handle)=false / DATABASE_NO_MORE_DATA(5126)
sql=UPDATE 'Struct' SET (id,name,number,timestamp,image)=
   (?1,?2,?3,STRFTIME('%s'),?5) WHERE rowid=4; / ok
DatabasePrepare(db,sql)=327681 / ok
DatabaseBind(handle,index,value)=true / ok
DatabaseBind(handle,index,value)=true / ok
DatabaseBind(handle,index,value)=true / ok
DatabaseBindArray(handle,index,value)=true / ok
DatabaseRead(handle)=false / DATABASE_NO_MORE_DATA(5126)
...
Finally, when getting binary data in two ways – incompatible, via the image string field as a result of
reading the entire DatabaseReadBind object (this is only done to visualize the sequence of bytes in the
log) and compatible, via DatabaseRead and DatabaseColumnBlob – we get different results: of course,
the second method is correct: the length and contents of the BLOB in 4096 bytes are restored.
sql=SELECT * FROM 'Struct' WHERE rowid=1; / ok
DatabasePrepare(db,sql)=393217 / ok
DatabaseReadBind(query,object)=true / ok
Length of string with Blob: 922
 7? 7? 7? 7? 7? 7?ɬ7? 6?ũ6?Ĩ5???5?¦5?Ĩ5? 6? 6? 7?ɬ7?ɬ7? 7? 7? 7? 7? 7? 7? 7? 7? 7? 7? 7??  ?? ?? ...
sql=SELECT * FROM 'Struct' WHERE rowid=1; / ok
DatabasePrepare(db,sql)=458753 / ok
DatabaseRead(query)=true / ok
Actual size of Blob: 4096
Summarizing the intermediate result of developing our own ORM wrapper, we present a generalized
scheme of its classes.

---

## Page 1893

Part 7. Advanced language tools
1 893
7.6 SQLite database
ORM Class Diagram (MQL5<->SQL)
7.6.1 3 Transactions
SQLite supports transactions – logically related sets of actions that can be performed either entirely or
not performed at all, which ensures the consistency of data in the database.
The concept of a transaction has a new meaning in the context of databases, different from what
we used to describe in trade transactions. A trade transaction means a separate operation on the
entities of a trading account, including orders, deals, and positions.
Transactions provide 4 main characteristics of database changes:
• Atomic (indivisible) – upon successful completion of the transaction, all the changes included in it
will get into the database, and in case of an error, nothing will get into it.
• Consistent – the current correct state of the base can only change to another correct state
(intermediate, according to application logic, states are excluded).
• Isolated – changes in the transaction of the current connection are not visible until the end of this
transaction in other connections to the same database and vice versa, changes from other
connections are not visible in the current connection while there is an incomplete transaction.
• Durable – changes from a successful transaction are guaranteed to be stored in the database.
The terms for these characteristics – Atomic, Consistent, Isolated, and Durable – form the acronym
ACID, well-known in database theory.
Even if the normal course of the program is interrupted due to a system failure, the database will retain
its working state.
Most often, the use of transactions is illustrated by the example of a banking system, in which funds are
transferred from the account of one client to the account of another. It should affect two records with

---

## Page 1894

Part 7. Advanced language tools
1 894
7.6 SQLite database
customer balances: in one, the balance is reduced by the amount of the transfer, and in the other, it is
increased. A situation where only one of these changes applies would upset the balance of bank
accounts: depending on which operation failed, the transferred amount could disappear or, conversely,
come from nowhere.
It is possible to give an example that is closer to trading practice but on the basis of the "opposite"
principle. The fact is that the system for accounting for orders, deals, and positions in MetaTrader 5 is
not transactional.
In particular, as we know from the chapter on Creating Expert Advisors, a triggered order (market or
pending), missing from the list of active ones, may not immediately be displayed in the list of positions.
Therefore, in order to analyze the actual result, it is necessary to implement in the MQL program the
expectation of updating (actualization) the trading environment. If the accounting system was based on
transactions, then the execution of an order, the registration of a transaction in history, and the
appearance of a position would be enclosed in a transaction and coordinated with each other. The
terminal developers have chosen a different approach: to return any modifications of the trading
environment as quickly and asynchronously as possible, and their integrity must be monitored by an
MQL program.
Any SQL command that changes the base (that is, in fact, everything except SELECT) will
automatically be wrapped in a transaction if this was not done explicitly beforehand.
The MQL5 API provides 3 functions for managing transactions: DatabaseTransactionBegin,
DatabaseTransactionCommit, and DatabaseTransactionRollback. All functions return true if successful or
false in case of an error.
bool DatabaseTransactionBegin(int database)
The DatabaseTransactionBegin function starts executing a transaction in the database with the
specified descriptor obtained from DatabaseOpen.
All subsequent changes made to the database are accumulated in the internal transaction cache and
do not get into the database until the DatabaseTransactionCommit function is called.
Transactions in MQL5 cannot be nested: if a transaction has already been started, then re-calling
DatabaseTransactionBegin will return an error flag and output a message to the log.
database error, cannot start a transaction within a transaction
DatabaseTransactionBegin(db)=false / DATABASE_ERROR(5601)
Respectively, you cannot try and complete the transaction multiple times.
bool DatabaseTransactionCommit(int database)
The function DatabaseTransactionCommit ends a transaction previously started in the database with
the specified handle and applies all accumulated changes (saves them). If an MQL program starts a
transaction but does not apply it before closing the database, all changes will be lost.
If necessary, the program can undo the transaction, and thus all changes since the beginning of the
transaction.
bool DatabaseTransactionRollback(int database)
The DatabaseTransactionRollback function performs a "rollback" of all actions included in the previously
started transaction for the database with the database handle.

---

## Page 1895

Part 7. Advanced language tools
1 895
7.6 SQLite database
Let's complete the DBSQLite class methods for working with transactions, taking into account the
restriction on their nesting, which we will calculate in the transaction variable. If it is 0, the begin
method starts a transaction by calling DatabaseTransactionBegin. All subsequent attempts to start a
transaction simply increase the counter. In the commit method, we decrement the counter, and when
it reaches 0 we call DatabaseTransactionCommit.
class DBSQLite
{
protected:
   int transaction;
   ...
public:
   bool begin()
   {
      if(transaction > 0)   // already in transaction
      {
         transaction++;     // keep track of the nesting level
         return true; 
      }
      return (bool)(transaction = PRTF(DatabaseTransactionBegin(handle)));
   }
   
   bool commit()
   {
      if(transaction > 0)
      {
         if(--transaction == 0) // outermost transaction
            return PRTF(DatabaseTransactionCommit(handle));
      }
      return false;
   }
   bool rollback()
   {
      if(transaction > 0)
      {
         if(--transaction == 0)
            return PRTF(DatabaseTransactionRollback(handle));
      }
      return false;
   }
};
Also, let's create the DBTransaction class, which will allow describing objects inside blocks (for example,
functions) that ensure the automatic start of a transaction with its subsequent application (or
cancellation) when the program exits the block.

---

## Page 1896

Part 7. Advanced language tools
1 896
7.6 SQLite database
class DBTransaction
{
   DBSQLite *db;
   const bool autocommit;
public:
   DBTransaction(DBSQLite &owner, const bool c = false): db(&owner), autocommit(c)
   {
      if(CheckPointer(db) != POINTER_INVALID)
      {
         db.begin();
      }
   }
   
   ~DBTransaction()
   {
      if(CheckPointer(db) != POINTER_INVALID)
      {
         autocommit ? db.commit() : db.rollback();
      }
   }
   
   bool commit()
   {
      if(CheckPointer(db) != POINTER_INVALID)
      {
         const bool done = db.commit();
         db = NULL;
         return done;
      }
      return false;
   }
};
The policy of using such objects eliminates the need to process various options for exiting a block
(function).
void DataFunction(DBSQLite &db)
{
   DBTransaction tr(db);
   DBQuery *query = db.prepare("UPDATE..."); // batch changes
   ... // base modification
   if(... /* error1 */) return;             // automatic rollback
   ... // base modification
   if(... /* error2 */) return;             // automatic rollback
   tr.commit();
}
For an object to automatically apply changes at any stage, pass true in the second parameter of its
constructor.

---

## Page 1897

Part 7. Advanced language tools
1 897
7.6 SQLite database
void DataFunction(DBSQLite &db)
{
   DBTransaction tr(db, true);
   DBQuery *query = db.prepare("UPDATE...");  // batch changes
   ... // base modification
   if(... /* condition1 */) return;           // automatic commit
   ... // base modification
   if(... /* condition2 */) return;           // automatic commit
   ...
}                                             // automatic commit
You can describe the DBTransaction object inside the loop and then, at each iteration, a separate
transaction will start and close.
A demonstration of transactions will be given in the section An example of searching for a trading
strategy using SQLite.
7.6.1 4 Import and export of database tables
MQL5 allows the export and import of individual database tables to/from CSV files. Export/import of the
entire database, as a file with SQL commands, is not provided.
long DatabaseImport(int database, const string table, const string filename, uint flags,
   const string separator, ulong skip_rows, const string comment_chars)
The DatabaseImport function imports data from the specified file into the table. The open database
descriptor and the table name are given by the first two parameters.
If tables named table does not exist, it will be created automatically. The names and types of fields in
the table will be recognized automatically based on the data contained in the file.  
The imported file can be not only a ready-made CSV file but also a ZIP archive with a CSV file. The
filename may contain a path. The file is searched relative to the MQL5/Files directory.
Valid flags that can be bitwise combined are described in the ENUM_DATABASE_IMPORT_FLAGS
enumeration:
• DATABASE_IMPORT_HEADER – the first line contains the names of the table fields
• DATABASE_IMPORT_CRLF – for line breaks, the CRLF character sequence is used
• DATABASE_IMPORT_APPEND – add data to an existing table
• DATABASE_IMPORT_QUOTED_STRINGS – string values in double quotes
• DATABASE_IMPORT_COMMON_FOLDER – common folder of terminals
Parameter separator sets the delimiter character in the CSV file.
Parameter skip_ rows skips the specified number of leading lines in the file.
Parameter comment_ chars contains the characters used in the file as a comment flag. Lines starting
with any of these characters will be considered comments and will not be imported.
The function returns the number of imported rows or -1  on error.

---

## Page 1898

Part 7. Advanced language tools
1 898
7.6 SQLite database
long DatabaseExport(int database, const string table_or_sql, const string filename, uint flags, const
string separator)
The DatabaseExport function exports a table or the result of an SQL query to a CSV file. The database
handle, as well as the table name or query text, are specified in the first two parameters.
If query results are exported, then the SQL query must begin with "SELECT" or "select". In other
words, a SQL query cannot change the database state; otherwise, DatabaseExport will end with an
error.
File filename name may contain a path inside the MQL5/Files directory of the current instance of the
terminal or the shared folder of terminals, depending on the flags.
The flags parameter allows you to specify a combination of flags that controls the format and location
of the file.
• DATABASE_EXPORT_HEADER – output a string with field names
• DATABASE_EXPORT_INDEX – display line numbers
• DATABASE_EXPORT_NO_BOM – do not insert a label BOM at the beginning of the file (BOM is
inserted by default)
• DATABASE_EXPORT_CRLF – use CRLF to break a line (LF by default)
• DATABASE_EXPORT_APPEND – append data to the end of an existing file (by default, the file is
overwritten), if the file does not exist, it will be created
• DATABASE_EXPORT_QUOTED_STRINGS – output string values in double quotes
• DATABASE_EXPORT_COMMON_FOLDER – CSV file will be created in the common folder of all
terminals MetaQuotes/Terminal/Common/File
Parameter separator specifies the column separator character. If it is NULL, then the tab character
'\t' will be used as a separator. The empty string "" is considered a valid delimiter, but the resulting
CSV file cannot be read as a table and it will be a set of rows.
Text fields in the database can contain newlines ('\r' or '\r\n' ) as well as the delimiter character
specified in the separator parameter. In this case, it is necessary to use the
DATABASE_EXPORT_QUOTED_STRINGS flag in the flags parameter. If this flag is present, all output
strings will be enclosed in double quotes, and if the string contains a double quote, it will be replaced by
two double quotes.
The function returns the number of exported records or a negative value in case of an error.
7.6.1 5 Printing tables and SQL queries to logs
If necessary, an MQL program can output the contents of a table or the results of an SQL query to a
log using the DatabasePrint function.
long DatabasePrint(int database, const string table_or_sql, uint flags)
The database handle is passed in the first parameter, followed by the table name or query text
(table_ or_ sql). The SQL query must start with "SELECT" or "select", i.e. it must not change the state
of the database. Otherwise, the DatabasePrint function will end with an error.
The flags parameter specifies a combination of flags that determine the formatting of the output.
·DATABASE_PRINT_NO_HEADER – do not display table column names (field names)

---

## Page 1899

Part 7. Advanced language tools
1 899
7.6 SQLite database
·DATABASE_PRINT_NO_INDEX – do not display line numbers
·DATABASE_PRINT_NO_FRAME – do not display a frame that separates the header and data
·DATABASE_PRINT_STRINGS_RIGHT – align strings to the right
If flags = 0, then columns and rows are displayed, the header and data are separated by a frame, and
the rows are aligned to the left.
The function returns the number of displayed records or -1  in case of an error.
We will use the function in the next section.
Unfortunately, the function does not allow an output of prepared queries with parameters. If there are
parameters, they will need to be embedded in the query text at the MQL5 level.
7.6.1 6 Example of searching for a trading strategy using SQLite
Let's try to use SQLite to solve practical problems. We will import structures into the MqlRates
database with the history of quotes and analyze them in order to identify patterns and search for
potential trading strategies. Of course, any chosen logic can also be implemented in MQL5, but SQL
allows you to do it in a different way, in many cases more efficiently and using many interesting built-in
SQL functions. The subject of the book, aimed at learning MQL5, does not allow going deep into this
technology, but we mention it as worthy of the attention of an algorithmic trader.
The script for converting quotes history into a database format is called DBquotesImport.mq5. In the
input parameters, you can set the prefix of the database name and the size of the transaction (the
number of records in one transaction).
input string Database = "MQL5Book/DB/Quotes";
input int TransactionSize = 1000;
To add MqlRates structures to the database using our ORM layer, the script defines an auxiliary
MqlRatesDB structure which provides the rules for binding structure fields to base columns. Since our
script only writes data to the database and does not read it from there, it does not need to be bound
using the DatabaseReadBind function, which would impose a restriction on the "simplicity" of the
structure. The absence of a constraint makes it possible to derive the MqlRatesDB structure from
MqlRates (and do not repeat the description of the fields).

---

## Page 1900

Part 7. Advanced language tools
1 900
7.6 SQLite database
struct MqlRatesDB: public MqlRates
{
   /* for reference:
   
      datetime time;
      double   open;
      double   high;
      double   low;
      double   close;
      long     tick_volume;
      int      spread;
      long     real_volume;
   */
   
   bool bindAll(DBQuery &q) const
   {
      return q.bind(0, time)
         && q.bind(1, open)
         && q.bind(2, high)
         && q.bind(3, low)
         && q.bind(4, close)
         && q.bind(5, tick_volume)
         && q.bind(6, spread)
         && q.bind(7, real_volume);
   }
   
   long rowid(const long setter = 0)
   {
 // rowid is set by us according to the bar time
      return time;
   }
};
   
DB_FIELD_C1(MqlRatesDB, datetime, time, DB_CONSTRAINT::PRIMARY_KEY);
DB_FIELD(MqlRatesDB, double, open);
DB_FIELD(MqlRatesDB, double, high);
DB_FIELD(MqlRatesDB, double, low);
DB_FIELD(MqlRatesDB, double, close);
DB_FIELD(MqlRatesDB, long, tick_volume);
DB_FIELD(MqlRatesDB, int, spread);
DB_FIELD(MqlRatesDB, long, real_volume);
The database name is formed from the prefix Database, name, and timeframe of the current chart on
which the script is running. A single table "MqlRatesDB" is created in the database with the field
configuration specified by the DB_FIELD macros. Please note that the primary key will not be
generated by the database, but is taken directly from the bars, from the time field (bar opening time).

---

## Page 1901

Part 7. Advanced language tools
1 901 
7.6 SQLite database
void OnStart()
{
   Print("");
   DBSQLite db(Database + _Symbol + PeriodToString());
   if(!PRTF(db.isOpen())) return;
   
   PRTF(db.deleteTable(typename(MqlRatesDB)));
   
   if(!PRTF(db.createTable<MqlRatesDB>(true))) return;
   ...
Next, using packages of TransactionSize bars, we request bars from the history and add them to the
table. This is a job of the helper function ReadChunk, called in a loop as long as there is data (the
function returns true) or the user won't stop the script manually. The function code is shown below.
   int offset = 0;
   while(ReadChunk(db, offset, TransactionSize) && !IsStopped())
   {
      offset += TransactionSize;
   }
Upon completion of the process, we ask the database for the number of generated records in the table
and output it to the log.
   DBRow *rows[];
   if(db.prepare(StringFormat("SELECT COUNT(*) FROM %s",
      typename(MqlRatesDB))).readAll(rows))
   {
      Print("Records added: ", rows[0][0].integer_value);
   }
}
The ReadChunk function looks as follows.

---

## Page 1902

Part 7. Advanced language tools
1 902
7.6 SQLite database
bool ReadChunk(DBSQLite &db, const int offset, const int size)
{
   MqlRates rates[];
   MqlRatesDB ratesDB[];
   const int n = CopyRates(_Symbol, PERIOD_CURRENT, offset, size, rates);
   if(n > 0)
   {
      DBTransaction tr(db, true);
      Print(rates[0].time);
      ArrayResize(ratesDB, n);
      for(int i = 0; i < n; ++i)
      {
         ratesDB[i] = rates[i];
      }
      
      return db.insert(ratesDB);
   }
   else
   {
      Print("CopyRates failed: ", _LastError, " ", E2S(_LastError));
   }
   return false;
}
It calls the built-in CopyRates function through which the rates bars array is filled. Then the bars are
transferred to the ratesDB array so that using just one statement db.insert(ratesDB) we could write
information to the database (we have formalized in MqlRatesDB how to do it correctly).
The presence of the DBTransaction object (with the automatic "commit" option enabled) inside the
block means that all operations with the array are "overlaid" with a transaction. To indicate progress,
during the processing of each block of bars, the label of the first bar is displayed in the log.
While the function CopyRates returns the data and their insertion into the database is successful, the
loop in OnStart continues with the shift of the numbers of the copied bars deep into the history. When
the end of the available history or the bar limit set in the terminal settings is reached, CopyRates will
return error 4401  (HISTORY_NOT_FOUND) and the script will exit.
Let's run the script on the EURUSD, H1  chart. The log should show something like this.
   db.isOpen()=true / ok
   db.deleteTable(typename(MqlRatesDB))=true / ok
   db.createTable<MqlRatesDB>(true)=true / ok
   2022.06.29 20:00:00
   2022.05.03 04:00:00
   2022.03.04 10:00:00
   ...
   CopyRates failed: 4401 HISTORY_NOT_FOUND
   Records added: 100000
We now have the base QuotesEURUSDH1 .sqlite, on which you can experiment to test various trading
hypotheses. You can open it in the MetaEditor to make sure that the data is transferred correctly.

---

## Page 1903

Part 7. Advanced language tools
1 903
7.6 SQLite database
Let's check one of the simplest strategies based on regularities in history. We will find the statistics of
two consecutive bars in the same direction, broken down by intraday time and day of the week. If there
is a tangible advantage for some combination of time and day of the week, it can be considered in the
future as a signal to enter the market in the direction of the first bar.
First, let's design an SQL query that requests quotes for a certain period and calculates the price
movement on each bar, that is, the difference between adjacent opening prices.
Since the time for bars is stored as a number of seconds (by the standards of datetime in MQL5 and,
concurrently, the "Unix epoch" of SQL), it is desirable to convert their display to a string for easy
reading, so let's start the SELECT query from the datetime field based on DATETIME function:
SELECT
   DATETIME(time, 'unixepoch') as datetime, open, ...
This field will not participate in the analysis and is given here only for the user. After that, the price is
displayed for reference, so that we can check the calculation of price increments by debug printing.
Since we are going to, if necessary, select a certain period from the entire file, the condition will
require the time field in "a pure form", and it should also be added to the request. In addition,
according to the planned analysis of quotes, we will need to isolate from the bar label its intraday time,
as well as the day of the week (their numbering corresponds to that adopted in MQL5, 0 is Sunday).
Let's call the last two columns of the query intraday and day, respectively, and the TIME and
STRFTIME functions are used to get them.
SELECT
   DATETIME(time, 'unixepoch') as datetime, open,
   time,
   TIME(time, 'unixepoch') AS intraday,
   STRFTIME('%w', time, 'unixepoch') AS day, ...
To calculate the price increment in SQL, you can use the LAG function. It returns the value of the
specified column with an offset of the specified number of rows. For example, LAG(X, 1 ) means getting
the X value in the previous entry, with the second parameter 1  that means the offset defaulting to 1 ,
i.e. it can be omitted to get the equivalent entry LAG(X). To get the value of the next entry, call
LAG(X,-1 ). In any case, when using LAG, an additional syntactic construction is required that specifies
the sorting order of records, in the simplest case, in the form of OVER(ORDER BY column).
Thus, to get the price increment between the opening prices of two neighboring bars, we write:
   ...
   (LAG(open,-1) OVER (ORDER BY time) - open) AS delta, ...
This column is predictive because it looks into the future.
We can reveal that two bars formed in the same direction by multiplying increments by them: positive
values indicate a consistent rise or fall:
   ...
   (LAG(open,-1) OVER (ORDER BY time) - open) * (open - LAG(open) OVER (ORDER BY time))
      AS product, ...
This indicator is chosen as the simplest to use in calculation: for real trading systems, you can choose
a more complex criterion.

---

## Page 1904

Part 7. Advanced language tools
1 904
7.6 SQLite database
To evaluate the profit generated by the system on the backtest, you need to multiply the direction of
the previous bar (which acts as an indicator of future movement) by the price increment on the next
bar. The direction is calculated in the column direction (using the SIGN function), for reference only.
The profit estimate in the estimate column is the product of the previous movement direction and the
increment of the next bar (delta): if the direction is preserved, we get a positive result (in points).
   ...
   SIGN(open - LAG(open) OVER (ORDER BY time)) AS direction,
   (LAG(open,-1) OVER (ORDER BY time) - open) * SIGN(open - LAG(open) OVER (ORDER BY time))
      AS estimate ...
In expressions in an SQL command, you cannot use AS aliases defined in the same command. That
is why we cannot determine estimate as delta * direction, and we have to repeat the calculation of
the product explicitly. However, we recall that columns delta and direction are not needed for
programmatic analysis and are added here only to visualize the table in front of the user.
At the end of the SQL command, we specify the table from which the selection is made, and the
filtering conditions for the backtest date range: two parameters "from" and "to".
...
FROM MqlRatesDB
WHERE (time >= ?1 AND time < ?2)
Optionally, we can add a constraint LIMIT?3 (and enter some small value, for example, 1 0) so that
visual verification of the query results at first does not force you to look through tens of thousands of
records.
You can check the operation of the SQL command using the DatabasePrint function, however, the
function, unfortunately, does not allow you to work with prepared queries with parameters. Therefore,
we will have to replace SQL parameter preparation '?n' with query string formatting using StringFormat
and substitute parameter values there. Alternatively, it would be possible to completely avoid
DatabasePrint and output the results to the log independently, line by line (through an array DBRow).
Thus, the final fragment of the request will turn into:
   ...
   WHERE (time >= %ld AND time < %ld)
   ORDER BY time LIMIT %d;
It should be noted that the datetime values in this query will be coming from MQL5 in the "machine"
format, i.e., the number of seconds since the beginning of 1 970. If we want to debug the same SQL
query in the MetaEditor, then it is more convenient to write the date range condition using date literals
(strings), as follows:
   WHERE (time >= STRFTIME('%s', '2015-01-01') AND time < STRFTIME('%s', '2021-01-01'))
Again, we need to use the STRFTIME function here (the '%s' modifier in SQL sets the transfer of the
specified date string to the "Unix epoch" label; the fact that '%s' resembles an MQL5 format string is
just a coincidence).
Save the designed SQL query in a separate text file DBQuotesIntradayLag.sql and connect it as a
resource to the test script of the same name, DBQuotesIntradayLag.mq5.

---

## Page 1905

Part 7. Advanced language tools
1 905
7.6 SQLite database
#resource "DBQuotesIntradayLag.sql" as string sql1
The first parameter of the script allows you to set a prefix in the name of the database, which should
already exist after launching DBquotesImport.mq5 on the chart with the same symbol and timeframe.
The subsequent inputs are for the date range and length limit of the debug printout to the log.
input string Database = "MQL5Book/DB/Quotes";
input datetime SubsetStart = D'2022.01.01';
input datetime SubsetStop = D'2023.01.01';
input int Limit = 10;
The table with quotes is known in advance, from the previous script.
const string Table = "MqlRatesDB";
In the OnStart function, we open the database and make sure that the quotes table is available.
void OnStart()
{
   Print("");
   DBSQLite db(Database + _Symbol + PeriodToString());
   if(!PRTF(db.isOpen())) return;
   if(!PRTF(db.hasTable(Table))) return;
   ...
Next, we substitute the parameters in the SQL query string. We pay attention not only to the
substitution of SQL parameters '?n' for format sequences but also double the percent symbols '%' first
because otherwise the function StringFormat will perceive them as its own commands, and will not miss
them in SQL.
   string sqlrep = sql1;
   StringReplace(sqlrep, "%", "%%");
   StringReplace(sqlrep, "?1", "%ld");
   StringReplace(sqlrep, "?2", "%ld");
   StringReplace(sqlrep, "?3", "%d");
   
   const string sqlfmt = StringFormat(sqlrep, SubsetStart, SubsetStop, Limit);
   Print(sqlfmt);
All these manipulations were required only to execute the request in the context of the DatabasePrint
function. In the working version of the analytical script, we would read the results of the query and
analyze them programmatically, bypassing formatting and calling DatabasePrint.
Finally, let's execute the SQL query and output the table with the results to the log.
   DatabasePrint(db.getHandle(), sqlfmt, 0);
}
Here is what we will see for 1 0 bars EURUSD,H1  at the beginning of 2022.

---

## Page 1906

Part 7. Advanced language tools
1 906
7.6 SQLite database
db.isOpen()=true / ok
db.hasTable(Table)=true / ok
      SELECT
         DATETIME(time, 'unixepoch') as datetime,
         open,
         time,
         TIME(time, 'unixepoch') AS intraday,
         STRFTIME('%w', time, 'unixepoch') AS day,
         (LAG(open,-1) OVER (ORDER BY time) - open) AS delta,
         SIGN(open - LAG(open) OVER (ORDER BY time)) AS direction,
         (LAG(open,-1) OVER (ORDER BY time) - open) * (open - LAG(open) OVER (ORDER BY time))
            AS product,
         (LAG(open,-1) OVER (ORDER BY time) - open) * SIGN(open - LAG(open) OVER (ORDER BY time))
            AS estimate
      FROM MqlRatesDB
      WHERE (time >= 1640995200 AND time < 1672531200)
      ORDER BY time LIMIT 10;
 #| datetime               open       time intraday day       delta dir       product      estimate
--+------------------------------------------------------------------------------------------------
 1| 2022-01-03 00:00:00 1.13693 1641168000 00:00:00 1  0.0003200098                                
 2| 2022-01-03 01:00:00 1.13725 1641171600 01:00:00 1  2.999999e-05  1  9.5999478e-09  2.999999e-05 
 3| 2022-01-03 02:00:00 1.13728 1641175200 02:00:00 1  -0.001060006  1 -3.1799748e-08  -0.001060006 
 4| 2022-01-03 03:00:00 1.13622 1641178800 03:00:00 1 -0.0003400007 -1  3.6040028e-07  0.0003400007 
 5| 2022-01-03 04:00:00 1.13588 1641182400 04:00:00 1  -0.001579991 -1  5.3719982e-07   0.001579991 
 6| 2022-01-03 05:00:00  1.1343 1641186000 05:00:00 1  0.0005299919 -1 -8.3739827e-07 -0.0005299919 
 7| 2022-01-03 06:00:00 1.13483 1641189600 06:00:00 1 -0.0007699937  1 -4.0809905e-07 -0.0007699937 
 8| 2022-01-03 07:00:00 1.13406 1641193200 07:00:00 1 -0.0002600149 -1  2.0020098e-07  0.0002600149 
 9| 2022-01-03 08:00:00  1.1338 1641196800 08:00:00 1   0.000510001 -1 -1.3260079e-07  -0.000510001 
10| 2022-01-03 09:00:00 1.13431 1641200400 09:00:00 1  0.0004800036  1  2.4480023e-07  0.0004800036 
...
It is easy to make sure that the intraday time of the bar is correctly allocated, as well as the day of the
week - 1 , which corresponds to Monday. You can also check the delta increment. The product and
estimate values are empty on the first row because they require the missing previous row to be
calculated.
Let's complicate our SQL query by grouping records with the same time of day combinations (intraday)
and day of the week (day), and calculating a certain target indicator that characterizes the success of
trading for each of these combinations. Let's take as such an indicator the average cell size product
divided by the standard deviation of the same products. The larger the average product of price
increments of neighboring bars, the greater the expected profit, and the smaller the spread of these
products, the more stable the forecast. The name of the indicator in the SQL query is obj ective.
In addition to the target indicator, we will also calculate the profit estimate (backtest_ profit) and profit
factor (backtest_ PF). We will estimate profit as the sum of price increments (estimate) for all bars in
the context of intraday time and day of the week (the size of the opening bar as a price increment is an
analog of the future profit in points per one bar). The profit factor is traditionally the quotient of
positive and negative increments.

---

## Page 1907

Part 7. Advanced language tools
1 907
7.6 SQLite database
   SELECT
      AVG(product) / STDDEV(product) AS objective,
      SUM(estimate) AS backtest_profit,
      SUM(CASE WHEN estimate >= 0 THEN estimate ELSE 0 END) /
         SUM(CASE WHEN estimate < 0 THEN -estimate ELSE 0 END) AS backtest_PF,
      intraday, day
   FROM
   (
      SELECT
         time,
         TIME(time, 'unixepoch') AS intraday,
         STRFTIME('%w', time, 'unixepoch') AS day,
         (LAG(open,-1) OVER (ORDER BY time) - open) AS delta,
         SIGN(open - LAG(open) OVER (ORDER BY time)) AS direction,
         (LAG(open,-1) OVER (ORDER BY time) - open) * (open - LAG(open) OVER (ORDER BY time))
            AS product,
         (LAG(open,-1) OVER (ORDER BY time) - open) * SIGN(open - LAG(open) OVER (ORDER BY time))
            AS estimate
      FROM MqlRatesDB
      WHERE (time >= STRFTIME('%s', '2015-01-01') AND time < STRFTIME('%s', '2021-01-01'))
   )
   GROUP BY intraday, day
   ORDER BY objective DESC
The first SQL query has become nested, from which we now accumulate data with an external SQL
query. Grouping by all combinations of time and day of the week provides an "extra" from GROUP BY
intraday, day. In addition, we have added sorting by target indicator (ORDER BY obj ective DESC) so that
the best options are at the top of the table.
In the nested query, we removed the LIMIT parameter, because the number of groups became
acceptable, much less than the number of analyzed bars. So, for H1  we get 1 20 options (24 * 5).
The extended query is placed in the text file DBQuotesIntradayLagGroup.sql, which in turn is connected
as a resource to the test script of the same name, DBQuotesIntradayLagGroup.mq5. Its source code
differs little from the previous one, so we will immediately show the result of its launch for the default
date range: from the beginning of 201 5 to the beginning of 2021  (excluding 2021  and 2022).

---

## Page 1908

Part 7. Advanced language tools
1 908
7.6 SQLite database
db.isOpen()=true / ok
db.hasTable(Table)=true / ok
   SELECT
      AVG(product) / STDDEV(product) AS objective,
      SUM(estimate) AS backtest_profit,
      SUM(CASE WHEN estimate >= 0 THEN estimate ELSE 0 END) /
         SUM(CASE WHEN estimate < 0 THEN -estimate ELSE 0 END) AS backtest_PF,
      intraday, day
   FROM
   (
      SELECT
         ...
      FROM MqlRatesDB
      WHERE (time >= 1420070400 AND time < 1609459200)
   )
   GROUP BY intraday, day
   ORDER BY objective DESC
  #|             objective       backtest_profit       backtest_PF intraday day
---+---------------------------------------------------------------------------
  1|      0.16713214428916     0.073200000000001  1.46040631486258 16:00:00 5   
  2|     0.118128291843983    0.0433099999999995  1.33678071539657 20:00:00 3   
  3|     0.103701251751617   0.00929999999999853  1.14148790506616 05:00:00 2   
  4|     0.102930330078208    0.0164399999999973   1.1932071923845 08:00:00 4   
  5|     0.089531492651001    0.0064300000000006  1.10167615433271 07:00:00 2   
  6|    0.0827628326995007 -8.99999999970369e-05 0.999601152226913 17:00:00 4   
  7|    0.0823433025146974    0.0159700000000012  1.21665988332657 21:00:00 1   
  8|    0.0767938336191962   0.00522999999999874  1.04226945769012 13:00:00 1   
  9|    0.0657741522256548    0.0162299999999986  1.09699976093712 15:00:00 2   
 10|    0.0635243373432768   0.00932000000000044  1.08294766820933 22:00:00 3
...   
110|   -0.0814131025461459   -0.0189100000000015 0.820605255668329 21:00:00 5   
111|   -0.0899571263478305   -0.0321900000000028 0.721250432975386 22:00:00 4   
112|   -0.0909772560603298   -0.0226100000000016 0.851161872161138 19:00:00 4   
113|   -0.0961794181717023  -0.00846999999999931 0.936377976414036 12:00:00 5   
114|    -0.108868074018582   -0.0246099999999998 0.634920634920637 00:00:00 5   
115|    -0.109368419185336   -0.0250700000000013 0.744496534855268 08:00:00 2   
116|    -0.121893581607986   -0.0234599999999998 0.610945273631843 00:00:00 3   
117|    -0.135416609546408   -0.0898899999999971 0.343437294573087 00:00:00 1   
118|    -0.142128458003631   -0.0255200000000018 0.681835182645536 06:00:00 4   
119|    -0.142196924506816   -0.0205700000000004 0.629769618430515 00:00:00 2   
120|     -0.15200009633513   -0.0301499999999988 0.708864426419475 02:00:00 1   
Thus, the analysis tells us that the 1 6-hour H1  bar on Friday is the best candidate to continue the
trend based on the previous bar. Next in preference is the Wednesday 20 o'clock bar. And so on.
However, it is desirable to check the found settings on the forward period.
To do this, we can execute the current SQL query not only on the "past" date range (in our test until
2021 ) but once more in the "future" (from the beginning of 2021 ). The results of both queries should
be joined (JOIN) by our groups (intraday, day). Then, while maintaining the sorting by the target
indicator, we will see in the adjacent columns the profit and profit factor for the same combinations of
time and day of the week, and how much they sank.
Here's the final SQL query (abbreviated):

---

## Page 1909

Part 7. Advanced language tools
1 909
7.6 SQLite database
SELECT * FROM
(
   SELECT
      AVG(product) / STDDEV(product) AS objective,
      SUM(estimate) AS backtest_profit,
      SUM(CASE WHEN estimate >= 0 THEN estimate ELSE 0 END) / 
         SUM(CASE WHEN estimate < 0 THEN -estimate ELSE 0 END) AS backtest_PF,
      intraday, day
   FROM
   (
      SELECT ...
      FROM MqlRatesDB
      WHERE (time >= STRFTIME('%s', '2015-01-01') AND time < STRFTIME('%s', '2021-01-01'))
   )
   GROUP BY intraday, day
) backtest
JOIN
(
   SELECT
      SUM(estimate) AS forward_profit,
      SUM(CASE WHEN estimate >= 0 THEN estimate ELSE 0 END) /
         SUM(CASE WHEN estimate < 0 THEN -estimate ELSE 0 END) AS forward_PF,
      intraday, day
   FROM
   (
      SELECT ...
      FROM MqlRatesDB
      WHERE (time >= STRFTIME('%s', '2021-01-01'))
   )
   GROUP BY intraday, day
) forward
USING(intraday, day)
ORDER BY objective DESC
The full text of the request is provided in the file DBQuotesIntradayBackAndForward.sql. It is connected
as a resource in the script DBQuotesIntradayBackAndForward.mq5.
By running the script with default settings, we get the following indicators (with abbreviations):
 #|          objective    backtest_profit    backtest_PF intraday day forward_profit     forward_PF
--+------------------------------------------------------------------------------------------------
 1|   0.16713214428916     0.073200000001  1.46040631486 16:00:00 5   0.004920000048  1.12852664576 
 2|  0.118128291843983    0.0433099999995  1.33678071539 20:00:00 3   0.007880000055    1.277856135 
 3|  0.103701251751617   0.00929999999853  1.14148790506 05:00:00 2   0.002210000082  1.12149532710 
 4|  0.102930330078208    0.0164399999973   1.1932071923 08:00:00 4   0.001409999969  1.07253086419 
 5|  0.089531492651001    0.0064300000006  1.10167615433 07:00:00 2  -0.009119999869 0.561749159058 
 6| 0.0827628326995007 -8.99999999970e-05 0.999601152226 17:00:00 4   0.009070000091  1.18809622563 
 7| 0.0823433025146974    0.0159700000012  1.21665988332 21:00:00 1    0.00250999999  1.12131464475 
 8| 0.0767938336191962   0.00522999999874  1.04226945769 13:00:00 1  -0.008490000055 0.753913043478 
 9| 0.0657741522256548    0.0162299999986  1.09699976093 15:00:00 2    0.01423999997  1.34979120609 
10| 0.0635243373432768   0.00932000000044  1.08294766820 22:00:00 3   -0.00456999993 0.828967065868
... 
So, the trading system with the best trading schedules found continues to show profit in the "future"
period, although not as large as on the backtest.

---

## Page 1910

Part 7. Advanced language tools
1 91 0
7.6 SQLite database
Of course, the considered example is only a particular case of a trading system. We could, for example,
find combinations of the time and day of the week when a reversal strategy works on neighboring bars,
or based on other principles altogether (analysis of ticks, calendar, portfolio of trading signals, etc.).
The bottom line is that the SQLite engine provides many convenient tools that would need to be
implemented in MQL5 on your own. To tell the truth, learning SQL takes time. The platform allows you
to choose the optimal combination of two technologies for efficient programming.
7.7 Development and connection of binary format libraries
In addition to the specialized types of MQL programs − Expert Advisors, indicators, scripts, and services
– the MetaTrader 5 platform allows you to create and connect independent binary modules with
arbitrary functionality, compiled as ex5 files or commonly used DLLs (Dynamic Link Library), standard
for Windows. These can be analytical algorithms, graphical visualization, network interaction with web
services, control of external programs, or the operating system itself. In any case, such libraries work
in the terminal not as independent MQL programs but in conjunction with a program of any of the above
4 types.
The idea of integrating the library and the main (parent) program is that the library exports certain
functions, i.e., declares them available for use from the outside, and the program imports their
prototypes. It is the description of prototypes – sets of names, lists of parameters, and return values –
that allows you to call these functions in the code without having their implementation.
Then, during the launch of the MQL program, the early dynamic linking is performed. This implies
loading the library after the main program and establishing correspondence between the imported
prototypes and the exported functions available in the library. Establishing one-to-one correspondences
by names, parameter lists, and return types is a prerequisite for successful loading. If no corresponding
exported implementation can be found for the import description of at least one function, the execution
of the MQL program will be canceled (it will end with an error at the startup stage).
Communication-component diagram of an MQL program with libraries

---

## Page 1911

Part 7. Advanced language tools
1 91 1 
7.7 Development and connection of binary format libraries
You cannot select an included library when starting an MQL program. This linking is set by the
developer when compiling the main program along with library imports. However, the user can manually
replace one ex5/dll file with another between program starts (provided that the prototypes of the
implemented exported functions match in the libraries). This can be used, for example, to switch the
user interface language if the libraries contain labeled string resources. However, libraries are most
often used as a commercial product with some know-how, which the author is not ready to distribute in
the form of open header files.
For programmers who have come to MQL5 from other environments and are already familiar with the
DLL technology, we would like to add a note about late dynamic linking, which is one of the advantages
of DLLs. Full dynamic connection of one MQL program (or DLL module) to another MQL program during
execution is impossible. The only similar action that MQL5 allows you to do "on the go" is linking an
Expert Advisor and an indicator via iCustom or IndicatorCreate, where the indicator acts as a
dynamically linked library (however, programmatic interaction with has to be done through the
indicators API, which means increased overhead for CopyBuffer, compared to direct function calls via
export/#import).
Note that in normal cases, when an MQL program is compiled from sources without importing external
functions, static linking is used, that is, the generated binary code directly refers to the called functions
since they are known at the time of compilation.
Strictly speaking, a library can also rely on other libraries, i.e., it can import some of the functions. In
theory, the chain of such dependencies can be even longer: for example, an MQL program includes
library A, library A uses library B, and library B, in turn, uses library C. However, such chains are
undesirable because they complicate the distribution and installation of the product, as well as make
identifying the causes of potential startup problems more difficult. Therefore, libraries are usually
connected directly to the parent MQL program.
In this chapter, we will describe the process of creating libraries in MQL5, exporting and importing
functions (including restrictions on the data types used in them), as well as connecting external (ready-
made) DLLs. DLL development is beyond the scope of this book.
7.7.1  Creation of ex5 libraries; export of functions 
To describe a library, add the #property library directive to the source code of the main (compiled)
module (usually, at the beginning of the file).
#property library
Specifying this directive in any other files included in the compilation process via #include has no effect.
The library property informs the compiler that the given ex5 file is a library: a mark about this is stored
in the header of the ex5 file.
A separate folder MQL5/Libraries is reserved for libraries in MetaTrader 5. You can organize a hierarchy
of nested folders in it, just like for other types of programs in MQL5.
Libraries do not directly participate in event handling, and therefore the compiler does not require the
presence of any standard handlers in the code. However, you can call the exported functions of the
library from the event handlers of the MQL program to which the library is connected.
To export a function from a library, just mark it with a special keyword export. This modifier must be
placed at the very end of the function header.

---

## Page 1912

Part 7. Advanced language tools
1 91 2
7.7 Development and connection of binary format libraries
result_type function_id ( [ parameter_type parameter_id
                          [ = default_value] ...] ) export
{
   ...
}
Parameters must be simple types or strings, structures with fields of such types, or their arrays.
Pointers and references are allowed for MQL5 object types (for restrictions on importing DLLs, see the
relevant section).
Let's see some examples. The parameter is a prime number:
double Algebraic2(const double x) export
{
   return x / sqrt(1 + x * x); 
}
The parameters are a pointer to an object and a reference to a pointer (allowing you to assign a pointer
inside the function).
class X
{
public:
   X() { Print(__FUNCSIG__); }
};
void setObject(const X *obj) export { ... }
void getObject(X *&obj) export { obj = new X(); }
The parameter is a structure:
struct Data
{
   int value;
   double data[];
   Data(): value(0) { }
   Data(const int i): value(i) { ArrayResize(data, i); }
};
   
void getRefStruct(const int i, Data &data) export { ... }
You can only export functions but not entire classes or structures. Some of these limitations can be
avoided with the help of pointers and references, which we will discuss in more detail later.
Function templates cannot be declared with the export keyword and in the #import directive.
The export modifier instructs the compiler to include the function in the table of exported functions
within the given ex5 executable. Thanks to this, such functions become available ("visible") from other
MQL programs, where they can be used after importing with a special directive #import.
All functions that are going to be exported must be marked with the export modifier. Although the main
program is not required to import all of them as it can only import the necessary ones.
If you forget to export a function but include it in the import directive in the main MQL program, then
when the latter is launched, an error will occur:

---

## Page 1913

Part 7. Advanced language tools
1 91 3
7.7 Development and connection of binary format libraries
cannot find 'function' in 'library.ex5'
unresolved import function call
A similar problem will arise if there are discrepancies in the description of the exported function and its
imported prototype. This can happen, for example, if you forget to recompile a library or main program
after making changes to the programming interface, which is usually described in a separate header
file.
Debugging libraries is not possible, so if necessary, you should have a helper script or another MQL
program that is built from the source codes of the library in debugger mode and can be executed
with breakpoints or step-by-step. Of course, this will require emulating calls to exported functions
using some real or artificial data.
For DLLs, the description of exported functions is done differently, depending on the programming
language in which they are created. Look for details in the documentation of your chosen
development environments.
Consider an example of a simple library MQL5/Libraries/MQL5Book/LibRand.mq5, from which several
functions are exported with different types of parameters and results. The library is designed to
generate random data:
• of numerical data with a pseudo-normal distribution
• of strings with random characters from the given sets (may be useful for passwords)
In particular, you can get one random number using the PseudoNormalValue function, in which the
expected value and variance are set as parameters.
double PseudoNormalValue(const double mean = 0.0, const double sigma = 1.0,
   const bool rooted = false) export
{
   // use ready-made sqrt for mass generation in a cycle in PseudoNormalArray
   const double s = !rooted ? sqrt(sigma) : sigma; 
   const double r = (rand() - 16383.5) / 16384.0; // [-1,+1] excluding borders
   const double x = -(log(1 / ((r + 1) / 2) - 1) * s) / M_PI * M_E + mean;
   return x;
}
The PseudoNormalArray function fills the array with random values in a given amount (n) and with the
required distribution.

---

## Page 1914

Part 7. Advanced language tools
1 91 4
7.7 Development and connection of binary format libraries
bool PseudoNormalArray(double &array[], const int n,
   const double mean = 0.0, const double sigma = 1.0) export
{
   bool success = true;
   const double s = sqrt(fabs(sigma)); // passing ready sqrt when calling PseudoNormalValue
   ArrayResize(array, n);
   for(int i = 0; i < n; ++i)
   {
      array[i] = PseudoNormalValue(mean, s, true);
      success = success && MathIsValidNumber(array[i]);
   }
   return success;
}
To generate one random string, we write the RandomString function, which "selects" from the supplied
set of characters (pattern) a given quantity (length) of arbitrary characters. When the pattern
parameter is blank (default), a full set of letters and numbers is assumed. Helper functions
StringPatternAlpha and StringPatternDigit are used to get it; these functions are also exportable (not
listed in the book, see the source code).
string RandomString(const int length, string pattern = NULL) export
{
   if(StringLen(pattern) == 0)
   {
      pattern = StringPatternAlpha() + StringPatternDigit();
   }
   const int size = StringLen(pattern);
   string result = "";
   for(int i = 0; i < length; ++i)
   {
      result += ShortToString(pattern[rand() % size]);
   }
   return result;
}
In general, to work with a library, it is necessary to publish a header file describing everything that
should be available in it from outside (and the details of the internal implementation can and should be
hidden). In our case, such a file is called MQL5Book/LibRand.mqh. In particular, it describes user-
defined types (in our case, the STRING_PATTERN enumeration) and function prototypes.
Although the exact syntax of the #import block is not known to us yet, this should not affect the clarity
of the declarations inside it: the headers of the exported functions are repeated here but without the
keyword export.

---

## Page 1915

Part 7. Advanced language tools
1 91 5
7.7 Development and connection of binary format libraries
enum STRING_PATTERN
{
   STRING_PATTERN_LOWERCASE = 1, // lowercase letters only
   STRING_PATTERN_UPPERCASE = 2, // capital letters only
   STRING_PATTERN_MIXEDCASE = 3  // both registers
};
   
#import "MQL5Book/LibRand.ex5"
string StringPatternAlpha(const STRING_PATTERN _case = STRING_PATTERN_MIXEDCASE);
string StringPatternDigit();
string RandomString(const int length, string pattern = NULL);
void RandomStrings(string &array[], const int n, const int minlength,
   const int maxlength, string pattern = NULL);
void PseudoNormalDefaultMean(const double mean = 0.0);
void PseudoNormalDefaultSigma(const double sigma = 1.0);
double PseudoNormalDefaultValue();
double PseudoNormalValue(const double mean = 0.0, const double sigma = 1.0,
   const bool rooted = false);
bool PseudoNormalArray(double &array[], const int n,
   const double mean = 0.0, const double sigma = 1.0);
#import
We will write a test script that uses this library in the next section, after studying the directive #import.
7.7.2 Including libraries; #import of functions
Functions are imported from compiled MQL5 modules (*.ex5 files) and from Windows dynamic library
modules (*.dll files). The module name is specified in the #import directive, followed by descriptions of
the imported function prototypes. Such a block must end with another #import directive, moreover, it
can be without a name and simply close the block itself, or the name of another library can be specified
in the directive, and thus the next import block begins at the same time. A series of import blocks
should always end with a directive without a library name.
In its simplest form, the directive looks like this:
#import "[path] module_name [.extension]"
  function_type function_name([parameter_list]);
  [function_type function_name([parameter_list]);]
   ... 
#import
The name of the library file can be specified without the extension: then the DLL is assumed by default.
Extension ex5 is required.
The name may be preceded by the library location path. By default, if there is no path, the libraries are
searched in the folder MQL5/Libraries or in the folder next to the MQL program where the library is
connected. Otherwise, different rules are applied to search for libraries depending on whether the type
is DLL or EX5. These rules are covered in a separate section.
Here is an example of sequential import blocks from two libraries:

---

## Page 1916

Part 7. Advanced language tools
1 91 6
7.7 Development and connection of binary format libraries
#import "user32.dll"
   int     MessageBoxW(int hWnd, string szText, string szCaption, int nType); 
   int     SendMessageW(int hWnd, int Msg, int wParam, int lParam); 
#import "lib.ex5" 
   double  round(double value); 
#import
With such directives, imported functions can be called from the source code in the same way as
functions defined directly in the MQL program itself. All technical issues with loading libraries and
redirecting calls to third-party modules are handled by the MQL program execution environment.
In order for the compiler to correctly issue the call to the imported function and organize the passing of
parameters, a complete description is required: with the result type, with all parameters, modifiers, and
default values, if they are present in the source.
Since the imported functions are outside of the compiled module, the compiler cannot check the
correctness of the passed parameters and return values. Any discrepancy between the format of the
expected and received data will result in an error during the execution of the program, and this may
manifest itself as a critical program stop, or unexpected behavior.
If the library could not be loaded or the called imported function was not found, the MQL program
terminates with a corresponding message in the log. The program will not be able to run until the
problem is resolved, for example, by modifying and recompiling, placing the required library in one of
the places along the search path, or allowing the use of the DLL (for DLLs only).
When sharing multiple libraries (doesn't matter if it's DLL or EX5), remember that they must have
different names, regardless of their location directories. All imported functions get a scope that
matches the name of the library file, that is, it is a kind of namespace, implicitly allocated for each
included library.
Imported functions can have any names, including those that match the names of built-in functions
(although this is not recommended). Moreover, it is possible to simultaneously import functions with the
same names from different modules. In such cases, the operation context permissions should be
applied to determine which function should be called.
For example:

---

## Page 1917

Part 7. Advanced language tools
1 91 7
7.7 Development and connection of binary format libraries
#import "kernel32.dll"
   int GetLastError();
#import "lib.ex5" 
   int GetLastError();
#import
  
class Foo
{
public: 
   int GetLastError() { return(12345); }
   void func() 
   { 
      Print(GetLastError());           // call a class method 
      Print(::GetLastError());         // calling the built-in (global) MQL5 function 
      Print(kernel32::GetLastError()); // function call from kernel32.d 
      Print(lib::GetLastError());      // function call from lib.ex5 
   }
};
   
void OnStart()
{
   Foo foo; 
   foo.func(); 
}
Let's see a simple example of the script LibRandTest.mq5, which uses functions from the EX5 library
created in the previous section.
#include <MQL5Book/LibRand.mqh>
In the input parameters, you can select the number of elements in the array of numbers, the
distribution parameters, as well as the step of the histogram, which we will calculate to make sure that
the distribution approximately corresponds to the normal law.
input int N = 10000;
input double Mean = 0.0;
input double Sigma = 1.0;
input double HistogramStep = 0.5;
input int RandomSeed = 0;
Initialization of the random number generator built into MQL5 (uniform distribution) is performed by the
value of the RandomSeed or, if 0 is left here, GetTickCount is picked (new at each start).
To build a histogram, we use MapArray and QuickSortStructT (we have already worked with them in the
sections on multicurrency indicators and about array sorting, respectively). The map will accumulate
counters of hitting random numbers in the cells of the histogram with a HistogramStep step.
#include <MQL5Book/MapArray.mqh>
#include <MQL5Book/QuickSortStructT.mqh>
To display a histogram based on the map, you need to be able to sort the map in key-value order. To
do this, we had to define a derived class.

---

## Page 1918

Part 7. Advanced language tools
1 91 8
7.7 Development and connection of binary format libraries
#define COMMA ,
   
template<typename K,typename V>
class MyMapArray: public MapArray<K,V>
{
public:
   void sort()
   {
      SORT_STRUCT(Pair<K COMMA V>, array, key);
   }
};
Note that the COMMA macro becomes an alternate representation of the comma character ',' and is
used when another SORT_STRUCT macro is called. If not for this substitution, the comma inside the
Pair<K,V> would be interpreted by the preprocessor as a normal macro parameter separator, as a
result of which 4 parameters would be received at the input of SORT_STRUCT instead of the expected
3 – this would cause a compilation error. The preprocessor knows nothing about the MQL5 syntax.
At the beginning of OnStart, after initialization of the generator, we check the receipt of a single
random string and an array of strings of different lengths.
void OnStart()
{
   const uint seed = RandomSeed ? RandomSeed : GetTickCount();
   Print("Random seed: ", seed);
   MathSrand(seed);
   
   // call two library functions: StringPatternDigit and RandomString
   Print("Random HEX-string: ", RandomString(30, StringPatternDigit() + "ABCDEF"));
   Print("Random strings:");
   string text[];
   RandomStrings(text, 5, 10, 20);         // 5 lines from 10 to 20 characters long
   ArrayPrint(text);
   ...
Next, we test normally distributed random numbers.

---

## Page 1919

Part 7. Advanced language tools
1 91 9
7.7 Development and connection of binary format libraries
   // call another library function: PseudoNormalArray
   double x[];
   PseudoNormalArray(x, N, Mean, Sigma);   // filled array x
   
   Print("Random pseudo-gaussian histogram: ");
   
   // take 'long' as key type, because 'int' has already been used for index access
   MyMapArray<long,int> map;
   
   for(int i = 0; i < N; ++i)
   {
 // value x[i] determines the cell of the histogram, where we increase the statistics
      map.inc((long)MathRound(x[i] / HistogramStep));
   }
   map.sort();                             // sort by key (i.e. by value)
   
   int max = 0;                            // searching for maximum for normalization
   for(int i = 0; i < map.getSize(); ++i)
   {
      max = fmax(max, map.getValue(i));
   }
   
   const double scale = fmax(max / 80, 1); // the histogram has a maximum of 80 symbols
   
   for(int i = 0; i < map.getSize(); ++i)  // print the histogram
   {
      const int p = (int)MathRound(map.getValue(i) / scale);
      string filler;
      StringInit(filler, p, '*');
      Print(StringFormat("%+.2f (%4d)",
         map.getKey(i) * HistogramStep, map.getValue(i)), " ", filler);
   }
Here is the result when run with default settings (timer randomization - each run will choose a new
seed).

---

## Page 1920

Part 7. Advanced language tools
1 920
7.7 Development and connection of binary format libraries
Random seed: 8859858
Random HEX-string: E58B125BCCDA67ABAB2F1C6D6EC677
Random strings:
"K4ZOpdIy5yxq4ble2" "NxTrVRl6q5j3Hr2FY" "6qxRdDzjp3WNA8xV"  "UlOPYinnGd36"      "6OCmde6rvErGB3wG" 
Random pseudo-gaussian histogram: 
-9.50 (   2) 
-8.50 (   1) 
-8.00 (   1) 
-7.00 (   1) 
-6.50 (   5) 
-6.00 (  10) *
-5.50 (  10) *
-5.00 (  24) *
-4.50 (  28) **
-4.00 (  50) ***
-3.50 ( 100) ******
-3.00 ( 195) ***********
-2.50 ( 272) ***************
-2.00 ( 510) ****************************
-1.50 ( 751) ******************************************
-1.00 (1029) *********************************************************
-0.50 (1288) ************************************************************************
+0.00 (1457) *********************************************************************************
+0.50 (1263) **********************************************************************
+1.00 (1060) ***********************************************************
+1.50 ( 772) *******************************************
+2.00 ( 480) ***************************
+2.50 ( 280) ****************
+3.00 ( 172) **********
+3.50 ( 112) ******
+4.00 (  52) ***
+4.50 (  43) **
+5.00 (  10) *
+5.50 (   8) 
+6.00 (   8) 
+6.50 (   2) 
+7.00 (   3) 
+7.50 (   1)
In this library, we have only exported and imported functions with built-in types. However, object
interfaces with structures, classes, and templates are much more interesting and more in demand from
a practical point of view. We will talk about the nuances of their use in libraries in a separate section.
When testing Expert Advisors and indicators in the tester, one should keep in mind an important
point related to libraries. Libraries required for the main tested MQL program are determined
automatically from the #import directives. However, if a custom indicator is called from the main
program, to which some library is connected, then it is necessary to explicitly indicate in the
program properties that it indirectly depends on a particular library. This is done with the directive: 
#property tester_library "path_library_name.extension"

---

## Page 1921

Part 7. Advanced language tools
1 921 
7.7 Development and connection of binary format libraries
7.7.3 Library file search order
If the library name is specified without a path or with a relative path, the search is performed according
to different rules depending on the type of library.
System libraries (DLL) are loaded according to the rules of the operating system. If the library is
already loaded (for example, by another Expert Advisor, or even from another client terminal launched
in parallel), then the call goes to the already loaded library. Otherwise, the search goes in the following
sequence:
1 .The folder from which the compiled EX5 program that imported the DLL was launched.
2. The MQL5/Libraries folder.
3. The folder where the running MetaTrader 5 terminal is located.
4. System folder (usually inside Windows).
5. Windows directory.
6. The current working folder of the terminal process (may be different from the terminal's location
folder).
7.Folders listed in the PATH system variable.
In the #import directives, it is not recommended to use a fully qualified loadable module name of the
form Drive:/Directory/FileName.dll.
If the DLL uses another DLL in its work, then in the absence of the second DLL, the first one will not be
able to load.
The search for an imported EX5 library is performed in the following sequence:
1 .Folder for launching the importing EX5 program.
2. Folder MQL5/Libraries of specific terminal instance.
3. Folder MQL5/Libraries in the common folder of all MetaTrader 5 terminals
(Common/MQL5/Libraries).
Before loading an MQL program, a general list of all EX5 library modules is formed, where the supported
modules are to be used both from the program itself and from libraries from this list. It's called a
dependency list and can become a very branched "tree".
For EX5 libraries, the terminal also provides a one-time download of reusable modules.
Regardless of the type of the library, each instance of it works with its own data related to the context
of the calling Expert Advisor, script, service, or indicator. Libraries are not a tool for shared access to
MQL5 variables or arrays.
EX5 libraries and DLLs run on the thread of the calling module.
There are no regular means to find in the library code where it was loaded from.
7.7.4 DLL connection specifics
The following entities cannot be passed as parameters into functions imported from a DLL:
• Classes (objects and pointers to them)
• Structures containing dynamic arrays, strings, classes, and other complex structures

---

## Page 1922

Part 7. Advanced language tools
1 922
7.7 Development and connection of binary format libraries
• Arrays of strings or the above complex objects
All simple type parameters are passed by value unless explicitly stated that they are passed by
reference. When passing a string, the buffer address of the copied string is passed; if the string is
passed by reference, then the buffer address of this particular string is passed to the function imported
from the DLL without copying.
When passing an array to DLL, the address of the data buffer beginning is always passed (regardless of
the AS_SERIES flag). The function inside the DLL knows nothing about the AS_SERIES flag, the passed
array is an array of unknown length, and an additional parameter is needed to specify its size.
When describing the prototype of an imported function, you can use parameters with default values.
When importing DLLs, you should give permission to use them in the properties of a specific MQL
program or in the general settings of the terminal. In this regard, in the Permissions section, we
presented the script EnvPermissions.mq5, which, in particular, has a function for reading the contents
of the Windows system clipboard using system DLLs. This function was provided optionally: its call was
commented out because we did not know how to work with libraries. Now, we will transfer it to a
separate script LibClipboard.mq5.
Running the script may prompt the user for confirmation (since DLLs are disabled by default for
security reasons). If necessary, enable the option in the dialog, on the tab with dependencies.
Header files are provided in the directory MQL5/Include/WinApi, which also includes #import directives
for much-needed system functions such as clipboard management (openclipboard, GetClipboardData,
and CloseClipboard), memory management (GlobalLock and GlobalUnlock), Windows windows, and many
others. We will include only two files: winuser.mqh and winbase.mqh. They contain the required import
directives and, indirectly, through the connection to windef.mqh, Windows term macros (HANDLE and
PVOID):
#define HANDLE  long
#define PVOID   long
   
#import "user32.dll"
...
int             OpenClipboard(HANDLE wnd_new_owner);
HANDLE          GetClipboardData(uint format);
int             CloseClipboard(void);
...
#import
   
#import "kernel32.dll"
...
PVOID           GlobalLock(HANDLE mem);
int             GlobalUnlock(HANDLE mem);
...
#import
In addition, we import the lstrcatW function from the kernel32.dll library because we are not satisfied
with its description in winbase.mqh provided by default: this gives the function a second prototype,
suitable for passing the PVOID value in the first parameter.

---

## Page 1923

Part 7. Advanced language tools
1 923
7.7 Development and connection of binary format libraries
#include <WinApi/winuser.mqh>
#include <WinApi/winbase.mqh>
   
#define CF_UNICODETEXT 13 // one of the standard exchange formats - Unicode text
#import "kernel32.dll"
string lstrcatW(PVOID string1, const string string2);
#import
The essence of working with the clipboard is to "capture" access to it using OpenClipboard, after which
you should get a data handle (GetClipboardData), convert it to a memory address (GlobalLock), and
finally copy the data from system memory to your variable (lstrcatW). Next, the occupied resources
are released in reverse order (GlobalUnlock and CloseClipboard).
void ReadClipboard()
{
   if(OpenClipboard(NULL))
   {
      HANDLE h = GetClipboardData(CF_UNICODETEXT);
      PVOID p = GlobalLock(h);
      if(p != 0)
      {
         const string text = lstrcatW(p, "");
         Print("Clipboard: ", text);
         GlobalUnlock(h);
      }
      CloseClipboard();
   }
}
Try copying the text to the clipboard and then running the script: the contents of the clipboard should
be logged. If the buffer contains an image or other data that does not have a textual representation,
the result will be empty.
Functions imported from a DLL follow the binary executable linking convention of Windows API
functions. To ensure this convention, compiler-specific keywords are used in the source text of
programs, such as, for example, _ _ stdcall in C or C++. These linking rules imply the following:
• The calling function (in our case, the MQL program) must see the prototype of the called (imported
from the DLL) function in order to correctly stack the parameters on the stack.
• The calling function (in our case, the MQL program) stacks parameters in reverse order, from right
to left – this is the order in which the imported function reads the parameters passed to it.
• Parameters are passed by value, except for those that are explicitly passed by reference (in our
case, strings).
• The imported function reads the parameters passed to it and clears the stack.
Here is another example of a script that uses a DLL – LibWindowTree.mq5. Its task is to go through the
tree of all terminal windows and get their class names (according to registration in the system using
WinApi) and titles. By windows here we mean the standard elements of the Windows interface, which
also include controls. This procedure can be useful for automating work with the terminal: emulating
button presses in windows, switching modes that are not available via MQL5, and so on.
To import the required system functions, let's include the header file WinUser.mqh that uses user32.dll.

---

## Page 1924

Part 7. Advanced language tools
1 924
7.7 Development and connection of binary format libraries
#include <WinAPI/WinUser.mqh>
You can get the name of the window class and its title using the functions GetClassNameW and
GetWindowTextW: they are called in the function GetWindowData.
void GetWindowData(HANDLE w, string &clazz, string &title)
{
   static ushort receiver[MAX_PATH];
   if(GetWindowTextW(w, receiver, MAX_PATH))
   {
      title = ShortArrayToString(receiver);
   }
   if(GetClassNameW(w, receiver, MAX_PATH))
   {
      clazz = ShortArrayToString(receiver);
   }
}
The 'W' suffix in function names means that they are intended for Unicode format strings (2 bytes per
character), which are the most commonly used today (the 'A' suffix for ANSI strings makes sense to
use only for backward compatibility with old libraries).
Given some initial handle to a Windows window, traversing up the hierarchy of its parent windows is
provided by the function TraverseUp: its operation is based on the system function GetParent. For each
found window, TraverseUp calls GetWindowData and outputs the resulting class name and title to the
log.
HANDLE TraverseUp(HANDLE w)
{
   HANDLE p = 0;
   while(w != 0)
   {
      p = w;
      string clazz, title;
      GetWindowData(w, clazz, title);
      Print("'", clazz, "' '", title, "'");
      w = GetParent(w);
   }
   return p;
}
Traversing deep into the hierarchy is performed by the function TraverseDown: the system function
FindWindowExW is used to enumerate child windows.

---

## Page 1925

Part 7. Advanced language tools
1 925
7.7 Development and connection of binary format libraries
HANDLE TraverseDown(const HANDLE w, const int level = 0)
{
   // request first child window (if any)
   HANDLE child = FindWindowExW(w, NULL, NULL, NULL);
   while(child)          // oop while there are child windows
   {
      string clazz, title;
      GetWindowData(child, clazz, title);
      Print(StringFormat("%*s", level * 2, ""), "'", clazz, "' '", title, "'");
      TraverseDown(child, level + 1);
      // requesting next child window
      child = FindWindowExW(w, child, NULL, NULL);
   }
   return child;
}
In the OnStart function, we find the main terminal window by traversing the windows up from the handle
of the current chart on which the script is running. Then we build the entire tree of terminal windows.
void OnStart()
{
   HANDLE h = TraverseUp(ChartGetInteger(0, CHART_WINDOW_HANDLE));
   Print("Main window handle: ", h);
   TraverseDown(h, 1);
}
We can also search for the required windows by class name and/or title, and therefore the main window
could be immediately obtained by calling FindWindowW, since its attributes are known.
   h = FindWindowW("MetaQuotes::MetaTrader::5.00", NULL); 
Here is an example log (snippet):

---

## Page 1926

Part 7. Advanced language tools
1 926
7.7 Development and connection of binary format libraries
 'AfxFrameOrView140su' ''
 'Afx:000000013F110000:b:0000000000010003:0000000000000006:00000000000306BA' 'EURUSD,H1'
 'MDIClient' ''
 'MetaQuotes::MetaTrader::5.00' '12345678 - MetaQuotes-Demo: Demo Account - Hedge - ...'
Main window handle: 263576
  'msctls_statusbar32' 'For Help, press F1'
  'AfxControlBar140su' 'Standard'
    'ToolbarWindow32' 'Timeframes'
    'ToolbarWindow32' 'Line Studies'
    'ToolbarWindow32' 'Standard'
  'AfxControlBar140su' 'Toolbox'
    'Afx:000000013F110000:b:0000000000010003:0000000000000000:0000000000000000' 'Toolbox'
      'AfxWnd140su' ''
        'ToolbarWindow32' ''
...
  'MDIClient' ''
    'Afx:000000013F110000:b:0000000000010003:0000000000000006:00000000000306BA' 'EURUSD,H1'
      'AfxFrameOrView140su' ''
        'Edit' '0.00'
    'Afx:000000013F110000:b:0000000000010003:0000000000000006:00000000000306BA' 'XAUUSD,Daily'
      'AfxFrameOrView140su' ''
        'Edit' '0.00'
    'Afx:000000013F110000:b:0000000000010003:0000000000000006:00000000000306BA' 'EURUSD,M15'
      'AfxFrameOrView140su' ''
        'Edit' '0.00'
7.7.5 Classes and templates in MQL5 libraries
Although the export and import of classes and templates are generally prohibited, the developer can
get around these restrictions by moving the description of the abstract base interfaces into the library
header file and passing pointers. Let's illustrate this concept with an example of a library that performs
a Hough transform of an image.
The Hough transform is an algorithm for extracting features of an image by comparing it with some
formal model (formula) described by a set of parameters.
The simplest Hough transform is the selection of straight lines on the image by converting them to
polar coordinates. With this processing, sequences of "filled" pixels, arranged more or less in a row,
form peaks in the space of polar coordinates at the intersection of a specific angle ("theta") of the
inclination of the straight line and its shift ("ro") relative to the center of coordinates.

---

## Page 1927

Part 7. Advanced language tools
1 927
7.7 Development and connection of binary format libraries
Hough transform for straight lines
Each of the three colored dots on the left (original) image leaves a trail in polar coordinate space (right)
because an infinite number of straight lines can be drawn through a point at different angles and
perpendiculars to the center. Each trace fragment is "marked" only once, with the exception of the red
mark: at this point, all three traces intersect and give the maximum response (3). Indeed, as we can
see in the original image, there is a straight line that goes through all three points. Thus, the two
parameters of the line are revealed by the maximum in polar coordinates.
We can use this Hough transform on price charts to highlight alternative support and resistance lines. If
such lines are usually drawn at individual extremes and, in fact, perform an analysis of outliers, then the
Hough transform lines can take into account all High or all Low prices, or even the distribution of tick
volumes within bars. All this allows you to get a more reasonable estimate of the levels.
Let's start with the header file LibHoughTransform.mqh. Since some abstract image supplies the initial
data for analysis, let's define the HoughImage interface template.
template<typename T>
interface HoughImage
{
   virtual int getWidth() const;
   virtual int getHeight() const;
   virtual T get(int x, int y) const;
};
All you need to know about the image when processing it is its dimensions and the content of each
pixel, which, for reasons of generality, is represented by the parametric type T. It is clear that in the
simplest case, it can be int or double.
Calling analytical image processing is a little more complicated. In the library, we need to describe the
class, the objects of which will be returned from a special factory function (in the form of pointers). It is
this function that should be exported from the library. Suppose, it is like this:

---

## Page 1928

Part 7. Advanced language tools
1 928
7.7 Development and connection of binary format libraries
template<typename T>
class HoughTransformDraft
{
public:
   virtual int transform(const HoughImage<T> &image, double &result[],
      const int elements = 8) = 0;
};
   
HoughTransformDraft<?> *createHoughTransform() export { ... } // Problem - template!
However, template types and template functions cannot be exported. Therefore, we will make an
intermediate non-template class HoughTransform, in which we will add a template method for the
image parameter. Unfortunately, template methods cannot be virtual, and therefore we will manually
dispatch calls inside the method (using dynamic_ cast), redirecting processing to a derived class with a
virtual method.
class HoughTransform
{
public:
   template<typename T>
   int transform(const HoughImage<T> &image, double &result[],
      const int elements = 8)
   {
      HoughTransformConcrete<T> *ptr = dynamic_cast<HoughTransformConcrete<T> *>(&this);
      if(ptr) return ptr.extract(image, result, elements);
      return 0;
   }
};
   
template<typename T>
class HoughTransformConcrete: public HoughTransform
{
public:
   virtual int extract(const HoughImage<T> &image, double &result[],
      const int elements = 8) = 0;
};
The internal implementation of the class HoughTransformConcrete will be written into the library file
MQL5/Libraries/MQL5Book/LibHoughTransform.mq5.

---

## Page 1929

Part 7. Advanced language tools
1 929
7.7 Development and connection of binary format libraries
#property library
   
#include <MQL5Book/LibHoughTransform.mqh>
   
template<typename T>
class LinearHoughTransform: public HoughTransformConcrete<T>
{
protected:
   int size;
   
public:
   LinearHoughTransform(const int quants): size(quants) { }
   ...
Since we are going to recalculate image points into space in new, polar, coordinates, a certain size
should be allocated for the task. Here we are talking about a discrete Hough transform since we
consider the original image as a discrete set of points (pixels), and we will accumulate the values of
angles with perpendiculars in cells (quanta). For simplicity, we will focus on the variant with a square
space, where the number of readings both in the angle and in the distance to the center is equal. This
parameter is passed to the class constructor.
template<typename T>
class LinearHoughTransform: public HoughTransformConcrete<T>
{
protected:
   int size;
   Plain2DArray<T> data;
   Plain2DArray<double> trigonometric;
   
   void init()
   {
      data.allocate(size, size);
      trigonometric.allocate(2, size);
      double t, d = M_PI / size;
      int i;
      for(i = 0, t = 0; i < size; i++, t += d)
      {
         trigonometric.set(0, i, MathCos(t));
         trigonometric.set(1, i, MathSin(t));
      }
   }
   
public:
   LinearHoughTransform(const int quants): size(quants)
   {
      init();
   }
   ...
To calculate the "footprint" statistics left by "filled" pixels in the transformed size space with
dimensions size by size, we describe the data array. The helper template class Plain2DArray (with type
parameter T) allows the emulation of a two-dimensional array of arbitrary sizes. The same class but

---

## Page 1930

Part 7. Advanced language tools
1 930
7.7 Development and connection of binary format libraries
with a parameter of type double is applied to the trigonometric table of pre-calculated values of sines
and cosines of angles. We will need the table to quickly map pixels to a new space.
The method for detecting the parameters of the most prominent straight lines is called extract. It takes
an image as input and must fill the output result array with found pairs of parameters of straight lines.
In the following equation:
y = a * x + b
the parameter a (slope, "theta") will be written to even numbers of the result array, and the b
parameter (indent, "ro") will be written to odd numbers of the array. For example, the first, most
noticeable straight line after the completion of the method is described by the expression:
y = result[0] * x + result[1];
For the second line, the indexes will increase to 2 and 3, respectively, and so on, up to the maximum
number of lines requested (lines). The result array size is equal to twice the number of lines.
template<typename T>
class LinearHoughTransform: public HoughTransformConcrete<T>
{
   ...
   virtual int extract(const HoughImage<T> &image, double &result[],
      const int lines = 8) override
   {
      ArrayResize(result, lines * 2);
      ArrayInitialize(result, 0);
      data.zero();
   
      const int w = image.getWidth();
      const int h = image.getHeight();
      const double d = M_PI / size;     // 180 / 36 = 5 degrees, for example
      const double rstep = MathSqrt(w * w + h * h) / size;
      ...
Nested loops over image pixels are organized in the straight line search block. For each "filled" (non-
zero) point, a loop through tilts is performed, and the corresponding pairs of polar coordinates are
marked in the transformed space. In this case, we simply call the method to increase the contents of
the cell by the value returned by the pixel: data.inc((int)r, i, v), but depending on the application and
type T, it may require more complex processing.

---

## Page 1931

Part 7. Advanced language tools
1 931 
7.7 Development and connection of binary format libraries
      double r, t;
      int i;
      for(int x = 0; x < w; x++)
      {
         for(int y = 0; y < h; y++)
         {
            T v = image.get(x, y);
            if(v == (T)0) continue;
   
            for(i = 0, t = 0; i < size; i++, t += d) // t < Math.PI
            {
               r = (x * trigonometric.get(0, i) + y * trigonometric.get(1, i));
               r = MathRound(r / rstep); // range [-range, +range]
               r += size; // [0, +2size]
               r /= 2;
   
               if((int)r < 0) r = 0;
               if((int)r >= size) r = size - 1;
               if(i < 0) i = 0;
               if(i >= size) i = size - 1;
   
               data.inc((int)r, i, v);
            }
         }
      }
      ...
In the second part of the method, the search for maximums in the new space is performed and the
output array result is filled.

---

## Page 1932

Part 7. Advanced language tools
1 932
7.7 Development and connection of binary format libraries
      for(i = 0; i < lines; i++)
      {
         int x, y;
         if(!findMax(x, y))
         {
            return i;
         }
   
         double a = 0, b = 0;
         if(MathSin(y * d) != 0)
         {
            a = -1.0 * MathCos(y * d) / MathSin(y * d);
            b = (x * 2 - size) * rstep / MathSin(y * d);
         }
         if(fabs(a) < DBL_EPSILON && fabs(b) < DBL_EPSILON)
         {
            i--;
            continue;
         }
         result[i * 2 + 0] = a;
         result[i * 2 + 1] = b;
      }
   
      return i;
   }
The findMax helper method (see the source code) writes the coordinates of the maximum value in the
new space to x and y variables, additionally overwriting the neighborhood of this place so as not to find
it again and again.
The LinearHoughTransform class is ready, and we can write an exportable factory function to spawn
objects.
HoughTransform *createHoughTransform(const int quants,
   const ENUM_DATATYPE type = TYPE_INT) export
{
   switch(type)
   {
   case TYPE_INT:
      return new LinearHoughTransform<int>(quants);
   case TYPE_DOUBLE:
      return new LinearHoughTransform<double>(quants);
   ...
   }
   return NULL;
}
Because templates are not allowed for export, we use the ENUM_DATATYPE enumeration in the second
parameter to vary the data type during conversion and in the original image representation.
To test the export/import of structures, we also described a structure with meta-information about the
transformation in a given version of the library and exported a function that returns such a structure.

---

## Page 1933

Part 7. Advanced language tools
1 933
7.7 Development and connection of binary format libraries
struct HoughInfo
{
   const int dimension; // number of parameters in the model formula
   const string about;  // verbal description
   HoughInfo(const int n, const string s): dimension(n), about(s) { }
   HoughInfo(const HoughInfo &other): dimension(other.dimension), about(other.about) { }
};
   
HoughInfo getHoughInfo() export
{
   return HoughInfo(2, "Line: y = a * x + b; a = p[0]; b = p[1];");
}
Various modifications of the Hough transforms can reveal not only straight lines but also other
constructions that correspond to a given analytical formula (for example, circles). Such modifications
will reveal a different number of parameters and carry a different meaning. Having a self-documenting
function can make it easier to integrate libraries (especially when there are a lot of them; note that our
header file contains only general information related to any library that implements this Hough
transform interface, and not just for straight lines).
Of course, this example of exporting a class with a single public method is somewhat arbitrary because
it would be possible to export the transformation function directly. However, in practice, classes tend to
contain more functionality. In particular, it is easy to add to our class the adjustment of the sensitivity
of the algorithm, the storage of exemplary patterns from lines for detecting signals checked on history,
and so on.
Let's use the library in an indicator that calculates support and resistance lines by High and Low prices
on a given number of bars. Thanks to the Hough transform and the programming interface, the library
allows you to display several of the most important such lines.
The source code of the indicator is in the file MQL5/Indicators/MQL5Book/p7/LibHoughChannel.mq5. It
also includes the header file LibHoughTransform.mqh, where we added the import directive.
#import "MQL5Book/LibHoughTransform.ex5"
HoughTransform *createHoughTransform(const int quants,
   const ENUM_DATATYPE type = TYPE_INT);
HoughInfo getHoughInfo();
#import
In the analyzed image, we denote by pixels the position of specific price types (OHLC) in quotes. To
implement the image, we need to describe the HoughQuotes class derived from Hough Image<int>.
We will provide for "painting" pixels in several ways: inside the body of the candles, inside the full range
of the candles, as well as directly in the highs and lows. All this is formalized in the PRICE_LINE
enumeration. For now, the indicator will use only HighHigh and LowLow, but this can be taken out in the
settings.

---

## Page 1934

Part 7. Advanced language tools
1 934
7.7 Development and connection of binary format libraries
class HoughQuotes: public HoughImage<int>
{
public:
   enum PRICE_LINE
   {
      HighLow = 0,   // Bar Range |High..Low|
      OpenClose = 1, // Bar Body |Open..Close|
      LowLow = 2,    // Bar Lows
      HighHigh = 3,  // Bar Highs
   };
   ...
In the constructor parameters and internal variables, we specify the range of bars for analysis. The
number of bars size determines the horizontal size of the image. For simplicity, we will use the same
number of readings vertically. Therefore, the price discretization step (step) is equal to the actual
range of prices (pp) for size bars divided by size. For the variable base, we calculate the lower limit of
prices that are subject to consideration in the indicated bars. This variable will be needed to bind the
construction of lines based on the found parameters of the Hough transform.
protected:
   int size;
   int offset;
   int step;
   double base;
   PRICE_LINE type;
   
public:
   HoughQuotes(int startbar, int barcount, PRICE_LINE price)
   {
      offset = startbar;
      size = barcount;
      type = price;
      int hh = iHighest(NULL, 0, MODE_HIGH, size, startbar);
      int ll = iLowest(NULL, 0, MODE_LOW, size, startbar);
      int pp = (int)((iHigh(NULL, 0, hh) - iLow(NULL, 0, ll)) / _Point);
      step = pp / size;
      base = iLow(NULL, 0, ll);
   }
   ...
Recall that the HoughImage interface requires the implementation of 3 methods: getWidth, getHeight,
and get. The first two are easy.

---

## Page 1935

Part 7. Advanced language tools
1 935
7.7 Development and connection of binary format libraries
   virtual int getWidth() const override
   {
      return size;
   }
   
   virtual int getHeight() const override
   {
      return size;
   }
The get method for getting "pixels" based on quotes returns 1  if the specified point falls within the bar
or cell range, according to the selected calculation method from PRICE_LINE. Otherwise, 0 is returned.
This method can be significantly improved by evaluating fractals, consistently increasing extremes, or
"round" prices with a higher weight (pixel fat).

---

## Page 1936

Part 7. Advanced language tools
1 936
7.7 Development and connection of binary format libraries
   virtual int get(int x, int y) const override
   {
      if(offset + x >= iBars(NULL, 0)) return 0;
   
      const double price = convert(y);
      if(type == HighLow)
      {
         if(price >= iLow(NULL, 0, offset + x) && price <= iHigh(NULL, 0, offset + x))
         {
            return 1;
         }
      }
      else if(type == OpenClose)
      {
         if(price >= fmin(iOpen(NULL, 0, offset + x), iClose(NULL, 0, offset + x))
         && price <= fmax(iOpen(NULL, 0, offset + x), iClose(NULL, 0, offset + x)))
         {
            return 1;
         }
      }
      else if(type == LowLow)
      {
         if(iLow(NULL, 0, offset + x) >= price - step * _Point / 2
         && iLow(NULL, 0, offset + x) <= price + step * _Point / 2)
         {
            return 1;
         }
      }
      else if(type == HighHigh)
      {
         if(iHigh(NULL, 0, offset + x) >= price - step * _Point / 2
         && iHigh(NULL, 0, offset + x) <= price + step * _Point / 2)
         {
            return 1;
         }
      }
      return 0;
   }
The helper method convert provides recalculation from pixel y coordinates to price values.
   double convert(const double y) const
   {
      return base + y * step * _Point;
   }
};
Now everything is ready for writing the technical part of the indicator. First of all, let's declare three
input variables to select the fragment to be analyzed, and the number of lines. All lines will be identified
by a common prefix.

---

## Page 1937

Part 7. Advanced language tools
1 937
7.7 Development and connection of binary format libraries
input int BarOffset = 0;
input int BarCount = 21;
input int MaxLines = 3;
   
const string Prefix = "HoughChannel-";
The object that provides the transformation service will be described as global: this is where the factory
function createHoughTransform is called from the library.
HoughTransform *ht = createHoughTransform(BarCount);
In the OnInit function, we just log the description of the library using the second imported function
getHoughInfo.
int OnInit()
{
   HoughInfo info = getHoughInfo();
   Print(info.dimension, " per ", info.about);
   return INIT_SUCCEEDED;
}
We will perform the calculation in OnCalculate once, at the opening of the bar.
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
{
   static datetime now = 0;
   if(now != iTime(NULL, 0, 0))
   {
      ... // see the next block
      now = iTime(NULL, 0, 0);
   }
   return rates_total;
}
The transformation calculation itself is run twice on a pair of images (highs and lows) formed by
different types of prices. In this case, the work is sequentially performed by the same object ht. If the
detection of straight lines was successful, we display them on the chart using the function DrawLine.
Because the lines are listed in the results array in descending order of importance, the lines are
assigned a decreasing weight.

---

## Page 1938

Part 7. Advanced language tools
1 938
7.7 Development and connection of binary format libraries
      HoughQuotes highs(BarOffset, BarCount, HoughQuotes::HighHigh);
      HoughQuotes lows(BarOffset, BarCount, HoughQuotes::LowLow);
      static double result[];
      int n;
      n = ht.transform(highs, result, fmin(MaxLines, 5));
      if(n)
      {
         for(int i = 0; i < n; ++i)
         {
            DrawLine(highs, Prefix + "Highs-" + (string)i,
               result[i * 2 + 0], result[i * 2 + 1], clrBlue, 5 - i);
         }
      }
      n = ht.transform(lows, result, fmin(MaxLines, 5));
      if(n)
      {
         for(int i = 0; i < n; ++i)
         {
            DrawLine(lows, Prefix + "Lows-" + (string)i,
               result[i * 2 + 0], result[i * 2 + 1], clrRed, 5 - i);
         }
      }
The DrawLine function is based on trend graphic objects (OBJ_TREND, see the source code).
When deinitializing the indicator, we delete the lines and the analytical object.
void OnDeinit(const int)
{
   AutoPtr<HoughTransform> destructor(ht);
   ObjectsDeleteAll(0, Prefix);
}
Before testing a new development, do not forget to compile both the library and the indicator.
Running the indicator with default settings gives something like this.

---

## Page 1939

Part 7. Advanced language tools
1 939
7.7 Development and connection of binary format libraries
Indicator with main lines for High/Low prices based on the Hough transform library
In our case, the test was successful. But what if you need to debug the library? There are no built-in
tools for this, so the following trick can be used. The library source test is conditionally compiled into a
debug version of the product, and the product is tested against the built library. Let's consider the
example of our indicator.
Let's provide the LIB_HOUGH_IMPL_DEBUG macro to enable the integration of the library source
directly into the indicator. The macro should be placed before including the header file.
#define LIB_HOUGH_IMPL_DEBUG
#include <MQL5Book/LibHoughTransform.mqh>
In the header file itself, we will overlay the import block from the binary standalone copy of the library
with preprocessor conditional compilation instructions. When the macro is enabled, another branch will
run, with the #include statement.
#ifdef LIB_HOUGH_IMPL_DEBUG
#include "../../Libraries/MQL5Book/LibHoughTransform.mq5"
#else
#import "MQL5Book/LibHoughTransform.ex5"
HoughTransform *createHoughTransform(const int quants,
   const ENUM_DATATYPE type = TYPE_INT);
HoughInfo getHoughInfo();
#import
#endif
In the library source file LibHoughTransform.mq5, inside the getHoughInfo function, we add output to
the log of information about the compilation method, depending on whether the macro is enabled or
disabled.

---

## Page 1940

Part 7. Advanced language tools
1 940
7.7 Development and connection of binary format libraries
HoughInfo getHoughInfo() export
{
#ifdef LIB_HOUGH_IMPL_DEBUG
   Print("inline library (debug)");
#else
   Print("standalone library (production)");
#endif
   return HoughInfo(2, "Line: y = a * x + b; a = p[0]; b = p[1];");
}
If in the indicator code, in the file LibHoughChannel.mq5  you uncomment the instruction #define
LIB_ HOUGH_ IMPL_ DEBUG, you can test the step-by-step image analysis.
7.7.6 Importing functions from .NET libraries
MQL5 provides a special service for working with .NET library functions: you can simply import the DLL
itself without specifying certain functions. MetaEditor automatically imports all the functions that you
can work with:
• Plain Old Data (POD) – structures that contain only simple data types;
• Public static functions whose parameters use only simple POD types and structures or their arrays.
Unfortunately, at the moment, it is not possible to see function prototypes as they are recognized
by MetaEditor.
For example, we have the following C# code of the Inc function of the TestClass class in the TestLib.dll
library:
public class TestClass
{ 
   public static void Inc(ref int x)
   {
      x++;
   }
}
Then, to import and call it, it is enough to write:
#import "TestLib.dll"
   
void OnStart()
{
   int x = 1;
   TestClass::Inc(x);
   Print(x);
}
After execution, the script will return the value of 2.
7.8 Projects
Software products, as a rule, are developed within the standard life cycle:
·Collection and addition of requirements

---

## Page 1941

Part 7. Advanced language tools
1 941 
7.8 Projects
·Design
·Development
·Testing
·Exploitation
As a result of constant improvement and expansion of functionality, it usually becomes necessary to
systematize source files, resources, and third-party libraries (here we mean not only binary format
libraries but, in a more general sense, any set of files, for example, headers). Even more, individual
programs are integrated into a common product that embodies an applied idea.
Structure and life cycle of the project
For example, when developing a trading robot, it is often necessary to connect ready-made or custom
indicators, the use of external machine learning algorithms implies writing a script for exporting quote
data and a script for re-importing trained models, and programs related to data exchange via the
Internet (for example, trading signals) may require web server and its settings in other programming
languages, at least for debugging and testing, if not for deploying a public service.
The whole complex of several interrelated products, together with their "dependencies" (which means
the used resources and libraries, written independently or taken from third-party sources), form a
software project.
When a program exceeds a certain size, its convenient and effective development is difficult without
special project management tools. This fully applies to programs based on MQL5, since many traders
use complex trading systems.
MetaEditor supports the concept of projects similar to other software packages. Currently, this
functionality is at the beginning of its development, and by the time the book is released, it will
probably change.
When working with projects in MQL5, keep in mind that the term "project" in the platform is used for
two different entities:

---

## Page 1942

Part 7. Advanced language tools
1 942
7.8 Projects
·Local project in the form of an mqproj file
·Folders in MQL5 cloud storage
A local project allows you to systematize and gather together all the information about source codes,
resources, and settings needed to build a particular MQL program. Such a project is only on your
computer and can refer to files from different folders.
The file with the extension mqproj  has a widely used, universal, JSON (JavaScript Object Notation) text
format. It is convenient, simple, and well-suited for describing data of any subject area: all information
is grouped into objects or arrays with named properties, with support for values of different types. All
this makes JSON conceptually very close to OOP languages; also it comes from object-oriented
JavaScript, as you can easily guess from the name.
Cloud storage operates on the basis of a version control system and collective work on software called
SVN (Subversion). Here, a project is a top-level folder inside the local directory MQL5/Shared Proj ects,
to which another folder is assigned, having the same name but located on the MQL5 Storage server.
Within a project folder, you can organize a hierarchy of subfolders. As the name suggests, network
projects can be shared with other developers and generally made public (the content can be
downloaded by anyone registered on mql5.com).
The system provides on-demand synchronization (using special user commands) between the folder
image in the cloud and on the local drive, and vice versa. You can both "pull" other people's project
changes to your computer, and "push" your edits to the cloud. Both the full folder image and selective
files can be synchronized, including, of course, mq5 files, mqh header files, multimedia, settings (set
files), as well as mqproj files. For more information about cloud storage, read the documentation of
MetaEditor and SVN systems.    
It is important to note that the existence of an mqproj file does not imply the creation of any cloud
project on its basis, just as the creation of a shared folder does not oblige you to use an mqproj
project.
At the time of this writing, an mqproj file can only describe the structure of one program, not several.
However, since such a requirement is common when developing complex projects, this functionality will
probably be added to MetaEditor in the future.
In this chapter, we will describe the main functions for creating and organizing mqproj projects and give
a series of examples.
7.8.1  General rules for working with local projects
A local project (mqproj file) can be created from the MetaEditor main menu or from the Navigator
context menu using commands New proj ect or New proj ect from source file. In the latter case, the file
must first be selected in Navigator or chosen in the Open dialog. As a result, the specified mq5 file will
be included in the project immediately. The first of the mentioned commands launches the MQL Wizard,
in which you should select the program type or an empty project option (source files can be added to it
later). The type of an MQL program for a project is chosen following the usual steps of the Wizard.
The project contains several logical sections which resemble a tree (hierarchy) with all the
components. They are displayed in the left panel of Navigator, in a separate tab Proj ect.

---

## Page 1943

Part 7. Advanced language tools
1 943
7.8 Projects
N avigator and indicator project properties
Immediately after creating the project or later by double-clicking on the root of the tree, a panel for
setting the MQL program properties opens in the right part of the window. The set of properties varies
depending on the type of program.
Most of the properties correspond to #property directives in the source code. These properties take
precedence: if you specify them in both the project and the source code, the values from the project
will be used.
Some developers may like to set properties interactively in a dialog rather than hardcoded in source
code. Also, you can use the same mq5 file in different projects and build versions of an MQL program
with different settings (without changing the source code).
Some properties are only available in a project. These include, for example, enabling/disabling
compilation optimizations and built-in divide-by-zero checks.
During project compilation, the system automatically analyzes dependencies, that is, the included
header files, resources, and so on. Dependencies appear in different branches of the project hierarchy.
In particular, header files from the standard MQL5/Include folders included in the #include directives
using angle brackets (<filename>), fall into Dependencies, and custom header files included with double
quotes (#include "filename") fall into the Headers section.
Additionally, the user can add files to the project that are related to the finished software product and
may be required for its normal operation or demonstration (for example, files with trained neural
network models) but are not directly embedded in the source code. For these purposes, you can use
the Settings and Files branch. Its context menu contains commands for adding a single file or an entire
directory to the project.
In particular, we will further consider examples of projects that will include not only client MQL
programs but also the server part.

---

## Page 1944

Part 7. Advanced language tools
1 944
7.8 Projects
Commands New file and New folder add a new element to the folder with the project file: such elements
are always searched relative to the project itself (in the mqproj file they are marked with the
relative_ to_ proj ect property equal to true, see further).
Commands Add an existing file and Add an existing folder select one or more elements from the existing
directory structure inside the MQL5 folder, and these elements inside the mqproj file are referenced
relative to the root MQL5 (the relative_ to_ proj ect property equals false).
The relative_ to_ proj ect property is just one of the few defined by the MetaTrader 5 developers to
represent a project in JSON format. Recall that as a result of editing the project (hierarchy and
properties), an mqproj-file of the JSON format is formed.
Here is what that file looks like for the project in the image above.
{
  "platform"    :"mt5",
  "program_type":"indicator",
  "copyright"   :"Copyright (c) 2015-2022, Marketeer",
  "link"        :"https:\/\/www.mql5.com\/en\/users\/marketeer",
  "version"     :"1.0",
  "description" :"Create 2 trend lines on highs and lows using Hough transform.",
  "optimize"    :"1",
  "fpzerocheck" :"1",
  "tester_no_cache":"0",
  "tester_everytick_calculate":"0",
  "unicode_character_set":"0",
  "static_libraries":"0",
  
  "indicator":
  {
    "window":"0"
  },
  
  "files":
  [
    {
      "path":"HoughChannel.mq5",
      "compile":true,
      "relative_to_project":true
    },
    {
      "path":"MQL5\\Include\\MQL5Book\\HoughTransform.mqh",
      "compile":false,
      "relative_to_project":false
    }
  ]
}
We will talk about the technical features of the JSON format in more detail in the following sections as
we will apply it in our demo projects.
It is important to note that all files referenced by the project are not stored inside the mqproj file,
and therefore copying to a new location or moving only the project file to another computer will not

---

## Page 1945

Part 7. Advanced language tools
1 945
7.8 Projects
restore it. To be able to migrate a project, set up a shared project for it and upload all the contents
of the project to the cloud. However, this may require a reorganization of the local file system
structure, as all components must be inside the shared project folder, while the mqproj format does
not require this.
7.8.2 Project plan of a web service for copying trades and signals
As an end-to-end demonstration project, which we will develop throughout this chapter, we will take a
simple, but at the same time quite technologically advanced product: a client-server copy trade
system. The client part will be MQL programs that communicate with the central part using the sockets
technology. Considering that MQL5 allows you to work only with client sockets, you will need to choose
an alternative platform for the socket server (more on that below). Thus, the project will require the
symbiosis of several different technologies and the use of many sections of the MQL5 API that we have
already studied, including application codes developed on their basis.
Thanks to the socket-based client-server architecture, the system can be used in different scenarios:
• for easy copying of trades between terminals on one computer;
• to establish a private (personal) communication channel between terminals on different computers,
including not only in the local network but also via the Internet;
• to organize a publicly open or closed signal service requiring registration;
• to monitor trading;
• to manage your own account remotely.
In all cases, client programs will act in 2 roles: a publisher (publisher, sender) and a subscriber
(recipient) of data.
We will not invent our own network protocol but will use the existing and popular WebSocket standard.
Their client implementation is built into all browsers, and we will need to repeat it (with a greater or
lesser degree of completeness) in MQL5. Of course, WebSocket support is also available for most
popular web servers. Therefore, in any case, our developments can not only be adapted to other
servers (if someone else suits) but also integrated with well-known sites that provide similar web
services. Here the whole point is to strictly follow the specification of their API, built on top of
WebSockets.
When developing software systems that are more complex than one standalone program, it is important
to draw up an action plan and, possibly, even design a technical project, including the structure of
modules, their interaction, and the sequence of coding.
So our plan includes:
1 .Theoretical analysis of the WebSocket protocol;
2. Selecting and installing a web server with the implementation of a WebSocket server;
3. Creating a simple echo server (sending a copy of incoming messages back to the client) to get
familiar with the technology;
4. Creating a simple client-side web page to test the functionality of the echo server from a browser;
5. Creating a simple chat server that sends messages to all connected clients, and a test web page
for it;
6. Creating a messaging server between identifiable providers and subscribers, and a test web client
for it;

---

## Page 1946

Part 7. Advanced language tools
1 946
7.8 Projects
7.Designing and implementing WebSockets in MQL5;
8. Creating a simple script as a client for an echo server;
9. Creating a simple Expert Advisor as a chat server client;
1 0.Finally, creating a trade copier in MQL5 which it will act as both an information provider (monitor
of account changes and status) and an information consumer (reproducing trades), depending on
the settings.
But before we start implementing the plan, we need to install a web server.
7.8.3 Nodejs based web server
To organize the server part of our projects, we need a web server. We will use the lightest and most
technologically advanced nodejs. Server-side scripts for it can be written in JavaScript, which is the
same language used in browsers for interactive web pages. This is convenient from the point of view of
unified writing of the client and server parts of the system; the client part of any web service, as a rule,
is required sooner or later, for example, for administration, registration, and displaying beautiful
statistics on the use of the service.
Anyone who knows MQL5 virtually knows JavaScript, so believe in yourselves. The main differences are
discussed in the sidebar.
MQL5 vs JavaScript
JavaScript is an interpreted language, unlike the compiled MQL5. For us as developers, this makes
life easier because we don't need a separate compilation phase to get a working program. Don't
worry about the efficiency of JavaScript: all JavaScript runtimes use JIT (just-in-time) compilation
of JavaScript on demand, i.e., the first time a module is accessed. This process occurs
automatically, implicitly, once per session, after which the script is executed in compiled form. 
MQL5 refers to languages with static typing, that is, when describing variables, we must explicitly
specify their type, and the compiler monitors type compatibility. In contrast, JavaScript is a
dynamically typed language: the type of a variable is determined by what value we put in it and can
change during the life of the variable. This provides flexibility but requires caution in order to avoid
unforeseen errors.
JavaScript is, in a sense, a more object-oriented language than MQL5, because almost all entities in
it are objects. For example, a function is also an object, and a class, as a descriptor of the
properties of objects, is also an object (of a prototype).
JavaScript itself "collects garbage", i.e., frees the memory allocated by the application program for
objects. In MQL5 we have to provide the timely call of delete for dynamic objects. 
The JavaScript syntax contains many convenient "abbreviations" for writing constructions that in
MQL5 have to be implemented in a longer way. For example, in order to pass a parameter pointing
to another function to a certain function in MQL5, we need to describe the type of such a pointer
using typedef, separately define a function that matches this prototype, and only then pass its
identifier as a parameter. In JavaScript, you can define the function you're pointing to (in its
entirety!) directly in the argument list instead of a pointer parameter.
If you are a web developer or already familiar with nodejs, you can skip the installation and
configuration steps.

---

## Page 1947

Part 7. Advanced language tools
1 947
7.8 Projects
You can download nodejs from the official site nodejs.org. Installation is available in different versions,
for example, using an installer or unpacking an archive. As a result of the installation, you will receive
an executable file in the specified directory node.exe and several supporting files and folders.
If nodejs was not added to the system path by the installer, this can be done for the current Windows
user by running the following command in the folder where nodejs is installed (where the file node.exe is
located):
setx PATH "%CD%"
Alternatively, you can edit the Windows environment variables from the system properties dialog
(Computer -> Properties -> Extra options -> Environment Variables; the specific dialog type depends on
the version of the operating system). In any case, in this way, we will ensure the ability to run nodejs
from any folder on the computer, which will be useful to us in the future.
You can check the health of nodejs by running the following commands (in the Windows command line):
node -v
npm version
The first command outputs the version of nodejs, and the second one outputs the version of an
important built-in nodejs service, the npm package manager.
A package is a ready-to-use module that adds specific functionality to nodejs. By itself, nodejs is very
small, and without packages, it would require a lot of routine coding.
The most requested packages are stored in a centralized repository on the web and can be downloaded
and installed on a specific copy of nodejs or globally (for all copies of nodejs if there are several on the
machine). Installing a package to a specific copy is done with the following command:
npm install <package name>
Run it in the folder where nodejs was installed. This command will place the package locally and will not
affect other copies of nodejs that already exist or may appear on the computer later on, with
unexpected edits.
We, in particular, need the ws package, which implements the WebSocket protocol. That is, you need
to run the command:
npm install ws
and wait for the process to complete. As a result, the folder <nodej s_ install_ path>/node_ modules/
should contain a new subfolder ws with the necessary content (you can look in the README.md file with
the description of the package to make sure it's a WebSocket protocol library).
The package contains implementations of both the server and the client. But instead of the latter, we
will write our own in MQL5.
All the functionality of the nodejs server is concentrated in the folder /node_ modules. It can be
compared in purpose with a standard folder MQL5/Include in MetaTrader 5. When writing application
programs in JavaScript, we will include or "import" the necessary modules in a special way, by analogy
with including mqh header files using the directive #include in MQL5.

---

## Page 1948

Part 7. Advanced language tools
1 948
7.8 Projects
7.8.4 Theoretical foundations of the WebSockets protocol
The WebSocket protocol is built on top of TCP/IP network connections, which are characterized by an
IP address (or a domain name that replaces it) and a port number. The HTTP/HTTPS protocol, with
which we have already practiced in the chapter on network functions, works based on the same
principle. There, the standard port numbers were 80 (for insecure connections) and 443 (for secure
connections). There is no dedicated port number for WebSocket, so web service providers can choose
any available number. All of our examples will use port 9000.
When specifying URLs as WebSocket protocol prefixes, we use ws (for non-secure connections) and wss
(for secure connections).
The WebSocket format is more efficient in terms of data transfer than HTTP as it uses much less
control data.
The initial connection establishment for a WebSocket service completely repeats an HTTP/HTTPS web
page request: you need to send a GET request with specially prepared headers. A feature of these
headers is the presence of lines:
Connection: Upgrade
Upgrade: websocket
as well as some additional lines that report the version of the WebSocket protocol and special randomly
generated strings. The keys involved in the "handshaking" procedure between the client and the server.
Sec-WebSocket-Key: ...
Sec-WebSocket-Version: 13
In practice, the "handshake" implies that the server checks the availability of those options that the
client requested, and in response with standard HTTP headers confirms the switch to WebSocket mode
or rejects it. The simplest reason for rejection can be if you are trying to connect via WebSockets to a
simple web server where the WebSocket server is not provided or the required version is not supported.
The current version of the WebSockets protocol is known under the symbolic name Hybi and number
1 3. An earlier and simpler version called Hixie may be useful for backward compatibility. In what
follows, we will only use Hybi, although a Hixie implementation is also included.
A successful connection is indicated by the following HTTP headers in the server response:
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: ...
The Sec-WebSocket-Accept field here is calculated and populated by the server based on the Sec-
WebSocket-Key to confirm compliance with the protocol. All this is regulated by the specification
RFC6455 and will be supported in our MQL programs as well.
For clarity, the procedure is shown in the following image:

---

## Page 1949

Part 7. Advanced language tools
1 949
7.8 Projects
Interaction between client and server via WebSocket protocol
After establishing a WebSocket connection, the client and server can exchange information packed into
special blocks: frames and messages. A message may consist of one or more frames. The frame size,
according to the specification, is limited to an astronomical number of 263 bytes
(9223372036854775807 ~ 9.22 exabytes!), but specific implementations may of course have more
mundane limits since this theoretical limit does not seem practical for sending in one packet.
At any time, the client or server can terminate the connection, having previously "politely said
goodbye" (see below) or by simply closing the network socket.  
Frames can be of different types as specified in their header (4 to 1 6 bytes long) that comes at the
beginning of each frame. For reference, let's list the operational codes (they are present in the first
byte of the header) and the purpose of frames of different types.
• 0 – continuation frame (inherits the properties of the previous frame);
• 1  – frame with text information;
• 2 – frame with binary information;
• 8 – frame request to close and confirmation of closing the connection (sent for "polite farewell");
• 9 – ping frame, can be periodically sent by either side to make sure the connection is physically
saved;
• 1 0 – pong frame, sent in response to a ping frame.
The last frame in a message is marked with a special bit in the header. Of course, when a message
consists of one frame, it is also the last one. The length of the payload is also passed in the header.
7.8.5 Server component of web services based on the WebSocket protocol
To organize a common server component of all projects, we will create a separate folder Web inside
MQL5/Experts/MQL5Book/p7/. Ideally, it would be convenient to place Web as a subfolder into Shared
Proj ects. The fact is that MQL5/Shared Proj ects is available in the standard distribution of MetaTrader 5

---

## Page 1950

Part 7. Advanced language tools
1 950
7.8 Projects
and reserved for cloud storage projects. Therefore, later, by using the functionality of shared projects,
it would be possible to upload all the files of our projects to the server (not only web files but also MQL
programs).
Later, when we create an mqproj file with MQL5 client programs, we will add all the files in this folder to
the project section Settings and Files, since all these files form an integral part of the project – the
server part.
Since a separate directory has been allocated for the project server, it is necessary to ensure the
possibility of importing modules from nodejs in this directory. By default, nodejs looks for modules in
the /node_ modules subfolder of the current directory, and we will run the server from the project.
Therefore, being in the folder where we will place the web files of the project, run the command:
mklink /j node_modules {drive:/path/to/folder/nodejs}/node_modules
As a result, a "symbolic" directory link called node_ modules will appear, pointing to the original folder
of the same name in the installed nodejs.
The easiest way to check the functionality of WebSockets is the echo service. Its model of operation is
to return any received message back to the sender. Let's consider how it would be possible to organize
such a service in a minimal configuration. An example is included in the file wsintro.j s.
First of all, we connect the package (module) ws, which provides WebSocket functionality for nodejs
and which we installed along with the web server.
// JavaScript
const WebSocket = require('ws');
The require function works similarly to the #include directive in MQL5, but additionally returns a module
object with the API of all files in the ws package. Thanks to this, we can call the methods and
properties of the WebSocket object. In this case, we need to create a WebSocket server on port 9000.
// JavaScript
const port = 9000;
const wss = new WebSocket.Server({ port: port });
Here we see the usual MQL5 constructor call by the new operator, but an unnamed object (structure)
is passed as a parameter, in which, as in a map, a set of named properties and their values can be
stored. In this case, only one property port is used, and its value is set equal to the (more precisely, a
constant) port variable described above. Basically, we can pass the port number (and other settings) on
the command line when running the script.
The server object gets into the wss variable. On success, we signal to the command line window that
the server is running (waiting for connections).
// JavaScript
console.log('listening on port: ' + port);
The console.log call is similar to the usual Print in MQL5. Also note that strings in JavaScript can be
enclosed not only in double quotes but also in single quotes, and even in backticks ` this is a
${template}text` , which adds some useful features.
Next, for the wss object, we assign a "connection" event handler, which refers to the connection of a
new client. Obviously, the list of supported object events is defined by the developers of the package, in
this case, the package ws that we use. All this is reflected in the documentation.

---

## Page 1951

Part 7. Advanced language tools
1 951 
7.8 Projects
The handler is bound by the on method, which specifies the name of the event and the handler itself.
// JavaScript
wss.on('connection', function(channel)
{
   ...
});
The handler is an unnamed (anonymous) function defined directly in the place where a reference
parameter is expected for the callback code to be executed on a new connection. The function is made
anonymous because it is used only here, and JavaScript allows such simplifications in the syntax. The
function has only one parameter which is the object of the new connection. We are free to choose the
name for the parameter ourselves, and in this case, it is channel.
Inside the handler, another handler should be set for the "message" event related to the arrival of a
new message in a specific channel.
// JavaScript
   channel.on('message', function(message)
   {
      console.log('message: ' + message);
      channel.send('echo: ' + message);
   });
   ...
It also uses an anonymous function with a single parameter, the received message object. We print it
to the console log for debugging. But the most important thing happens in the second line: by calling
channel.send, we send a response message to the client.
To complete the picture, let's add our own welcome message to the "connection" handler. When
complete, it looks like this:
// JavaScript
wss.on('connection', function(channel)
{
   channel.on('message', function(message)
   {
      console.log('message: ' + message);
      channel.send('echo: ' + message);
   });
   console.log('new client connected!');
   channel.send('connected!');
});
It's important to understand that while binding the "message" handler is higher in the code than
sending the "hello", the message handler will be called later, and only if the client sends a message.
We have reviewed a script outline for organizing an echo service. However, it would be good to test it.
This can be done in the most efficient way by using a regular browser, but this will require complicating
the script slightly: turning it into the smallest possible web server that returns a web page with the
smallest possible WebSocket client.

---

## Page 1952

Part 7. Advanced language tools
1 952
7.8 Projects
Echo service and test web page
The echo server script that we will now look at is in the file wsecho.j s. One of the main points is that it
is desirable to support not only open protocols on the server http/ws but also protected protocols
https/wss. This possibility will be provided in all our examples (including clients based on MQL5), but for
this, you need to perform some actions on the server.
You should start with a couple of files containing encryption keys and certificates. The files are usually
obtained from authorized sources, i.e. certifying centers, but for informational purposes, you can
generate the files yourself. Of course, they cannot be used on public servers, and pages with such
certificates will cause warnings in any browser (the page icon to the left of the address bar is
highlighted in red).
The description of the device of certificates and the process of generating them on their own is beyond
the scope of the book, but two ready-made files are included in the book: MQL5Book.crt and
MQL5Book.key (there are other extensions) with a limited duration. These files must be passed to the
constructor of the web server object in order for the server to work over the HTTPS protocol.
We will pass the name of the certificate files in the script launch command line. For example, like this:
node wsecho.js MQL5Book
If you run the script without an additional parameter, the server will work using the HTTP protocol.
node wsecho.js
Inside the script, command line arguments are available through the built-in object process.argv, and
the first two arguments always contain, respectively, the name of the server node.exe and the name of
the script to run (in this case, wsecho.j s), so we discard them by the splice method.
// JavaScript
const args = process.argv.slice(2);
const secure = args.length > 0 ? 'https' : 'http';
Depending on the presence of the certificate name, the secure variable gets the name of the package
that should be loaded next to create the server: https or http. In total, we have 3 dependencies in the
code:
// JavaScript
const fs = require('fs');
const http1 = require(secure);
const WebSocket = require('ws');
We already know all about the ws package; the https and http packages provide a web server
implementation, and the built-in fs package provides work with the file system.
Web server settings are formatted as the options object. Here we see how the name of the certificate
from the command line is substituted in strings with slash quotes using the expression ${args[0]}. Then
the corresponding pair of files is read by the method fs.readFileSync.

---

## Page 1953

Part 7. Advanced language tools
1 953
7.8 Projects
// JavaScript
const options = args.length > 0 ?
{
   key : fs.readFileSync(`${args[0]}.key`),
   cert : fs.readFileSync(`${args[0]}.crt`)
} : null;
The web server is created by calling the createServer method, to which we pass the options object and
an anonymous function – an HTTP request handler. The handler has two parameters: the req object
with an HTTP request and the res object with which we should send the response (HTTP headers and
web page).
// JavaScript
http1.createServer(options, function (req, res)
{
   console.log(req.method, req.url);
   console.log(req.headers);
   
   if(req.url == '/') req.url = "index.htm";
   
   fs.readFile('./' + req.url, (err, data) =>
   {
      if(!err)
      {
         var dotoffset = req.url.lastIndexOf('.');
         var mimetype = dotoffset == -1 ? 'text/plain' :
         {
            '.htm' : 'text/html',
            '.html' : 'text/html',
            '.css' : 'text/css',
            '.js' : 'text/javascript'
         }[ req.url.substr(dotoffset) ];
         res.setHeader('Content-Type',
            mimetype == undefined ? 'text/plain' : mimetype);
         res.end(data);
      }
      else
      {
         console.log('File not fount: ' + req.url);
         res.writeHead(404, "Not Found");
         res.end();
      }
  });
}).listen(secure == 'https' ? 443 : 80);
The main index page (and the only one) is index.htm (to be written now). In addition, the handler can
send js and css files, which will be useful to us in the future. Depending on whether protected mode is
enabled, the server is started by calling the method listen on standard ports 443 or 80 (change to
others if these are already taken on your computer).
To accept connections on port 9000 for web sockets, we need to deploy another web server instance
with the same options. But in this case, the server is there for the sole purpose of handling an HTTP
request to "upgrade" the connection up to the Web Sockets protocol.

---

## Page 1954

Part 7. Advanced language tools
1 954
7.8 Projects
// JavaScript
const server = new http1.createServer(options).listen(9000);
server.on('upgrade', function(req, socket, head)
{
   console.log(req.headers); // TODO: we can add authorization!
});
Here, in the "upgrade" event handler, we accept any connections that have already passed the
handshake and print the headers to the log, but potentially we could request user authorization if we
were doing a closed (paid) service.
Finally, we create a WebSocket server object, as in the previous introductory example, with the only
difference being that a ready-made web server is passed to the constructor. All connecting clients are
counted and welcomed by sequence number.
// JavaScript
var count = 0;
   
const wsServer = new WebSocket.Server({ server });
wsServer.on('connection', function onConnect(client)
{
   console.log('New user:', ++count);
   client.id = count; 
   client.send('server#Hello, user' + count);
   
   client.on('message', function(message)
   {
      console.log('%d : %s', client.id, message);
      client.send('user' + client.id + '#' + message);
   });
   
   client.on('close', function()
   {
      console.log('User disconnected:', client.id);
   });
});
For all events, including connection, disconnection, and message, debug information is displayed in the
console.
Well, the web server with web socket server support is ready. Now we need to create a client web page
index.htm for it.

---

## Page 1955

Part 7. Advanced language tools
1 955
7.8 Projects
<!DOCTYPE html>
<html>
  <head>
  <title>Test Server (HTTP[S]/WS[S])</title>
  </head>
  <body>
    <div>
      <h1>Test Server (HTTP[S]/WS[S])</h1>
      <p><label>
         Message: <input id="message" name="message" placeholder="Enter a text">
      </label></p>
      <p><button>Submit</button> <button>Close</button></p>
      <p><label>
         Echo: <input id="echo" name="echo" placeholder="Text from server">
      </label></p>
    </div>
  </body>
  <script src="wsecho_client.js"></script>
</html>
The page is a form with a single input field and a button for sending a message.
Echo service web page on WebSocket
The page uses the wsecho_ client.j s script, which provides websocket client response. In browsers, web
sockets are built in as "native" JavaScript objects, so you don't need to connect anything external: just
call the constructor web socket with the desired protocol and port number.
// JavaScript
const proto = window.location.protocol.startsWith('http') ?
              window.location.protocol.replace('http', 'ws') : 'ws:';
const ws = new WebSocket(proto + '//' + window.location.hostname + ':9000');
The URL is formed from the address of the current web page (window.location.hostname), so the web
socket connection is made to the same server.
Next, the ws object allows you to react to events and send messages. In the browser, the open
connection event is called "open"; it is connected via the onopen property. The same syntx, slightly
different from the server implementation, is also used for the new message arrival event – the handler
for it is assigned to the onmessage property.

---

## Page 1956

Part 7. Advanced language tools
1 956
7.8 Projects
// JavaScript
ws.onopen = function()
{
   console.log('Connected');
};
   
ws.onmessage = function(message)
{
   console.log('Message: %s', message.data);
   document.getElementById('echo').value = message.data; 
};
The text of the incoming message is displayed in the form element with the id "echo". Note that the
message event object (handler parameter) is not the message which is available in the data property.
This is an implementation feature in JavaScript.
The reaction to the form buttons is assigned using the addEventListener method for each of the two
button tag objects. Here we see another way of describing an anonymous function in JavaScript:
parentheses with an argument list that can be empty, and the body of the function after the arrow can
be (arguments) => { ... }.
// JavaScript
const button = document.querySelectorAll('button'); // request all buttons
// button "Submit"  
button[0].addEventListener('click', (event) =>
{
   const x = document.getElementById('message').value;
   if(x) ws.send(x);
});
// button "close"
button[1].addEventListener('click', (event) =>
{
   ws.close();
   document.getElementById('echo').value = 'disconnected';
   Array.from(document.getElementsByTagName('button')).forEach((e) =>
   {
      e.disabled = true;
   });
});
To send messages, we call the ws.send method, and to close the connection we call the ws.close
method.
This completes the development of the first example of client-server scripts for demonstrating the echo
service. You can run wsecho.j s using one of the commands shown earlier, and then open in your browser
the page at http://localhost or https://localhost (depending on server settings). After the form appears
on the screen, try chatting with the server and make sure the service is running.
Gradually complicating this example, we will pave the way for the web service for copying trading
signals. But the next step will be a chat service, the principle of which is similar to the service of trading
signals: messages from one user are transmitted to other users.

---

## Page 1957

Part 7. Advanced language tools
1 957
7.8 Projects
Chat service and test web page
The new server script is called wschat.j s, and it repeats a lot from wsecho.j s. Let's list the main
differences. In the web server HTTP request handler, change the initial page from index.htm to
wschat.htm.
// JavaScript
http1.createServer(options, function (req, res)
{
   if(req.url == '/') req.url = "wschat.htm";
   ...
});
To store information about users connected to the chat, we will describe the clients map array. Map is
a standard JavaScript associative container, into which arbitrary values can be written using keys of an
arbitrary type, including objects.
// JavaScript
const clients = new Map();                    // added this line
var count = 0;
In the new user connection event handler, we will add the client object, received as a function
parameter, into the map under the current client sequence number.
// JavaScript
wsServer.on('connection', function onConnect(client)
{
   console.log('New user:', ++count);
   client.id = count; 
   client.send('server#Hello, user' + count);
   clients.set(count, client);                // added this line
   ...
Inside the onConnect function, we set a handler for the event about the arrival of a new message for a
specific client, and it is inside the nested handler that we send messages. However, this time we loop
through all the elements of the map (that is, through all the clients) and send the text to each of them.
The loop is organized with the forEach method calls for an array from the map, and the next anonymous
function that will be performed for each element (elem) is passed to the method in place. The example
of this loop clearly demonstrates the functional-declarative programming paradigm that prevails in
JavaScript (in contrast to the imperative approach in MQL5).
// JavaScript
   client.on('message', function(message)
   {
      console.log('%d : %s', client.id, message);
      Array.from(clients.values()).forEach(function(elem) // added a loop
      {
         elem.send('user' + client.id + '#' + message);
      });
   });
It is important to note that we send a copy of the message to all clients, including the original author.
It could be filtered out, but for debugging purposes, it's better to have confirmation that the message
was sent.

---

## Page 1958

Part 7. Advanced language tools
1 958
7.8 Projects
The last difference from the previous echo service is that when a client disconnects, it needs to be
removed from the map.
// JavaScript
   client.on('close', function()
   {
      console.log('User disconnected:', client.id);
      clients.delete(client.id);                   // added this line
   });
Regarding the replacement of the page index.htm by wschat.htm, here we added a "field" to display the
author of the message (origin) and connected a new browser script wschat_ client.j s. It parses the
messages (we use the '#' symbol to separate the author from the text) and fills in the form fields with
the information received. Since nothing has changed from the point of view of the WebSocket protocol,
we will not provide the source code.
Chat service webpage on WebSocket
You can start nodejs with the wschat.j s chat server and then connect to it from several browser tabs.
Each connection gets a unique number displayed in the header. Text from the Message field is sent to
all clients upon the click on Submit. Then, the client forms show both the author of the message (label
at the bottom left) and the text itself (field at the bottom center).
So, we have made sure that the web server with web socket support is ready. Let's turn to writing the
client part of the protocol in MQL5.
7.8.6 WebSocket protocol in MQL5
We have previously looked at Theoretical foundations of the WebSockets protocol. The complete
specification is quite extensive, and a detailed description of its implementation would require a lot of
space and time. Therefore, we present the general structure of ready-made classes and their
programming interfaces. All files are located in the directory MQL5/Include/MQL5Book/ws/.
• wsinterfaces.mqh – general abstract description of all interfaces, constants, and types;
• wstransport.mqh – MqlWebSocketTransport class that implements the IWebSocketTransport low-
level network data transfer interface based on MQL5 socket functions;
• wsframe.mqh – WebSocketFrame and WebSocketFrameHixie classes that implement the
IWebSocketFrame interface, which hides the algorithms for generating (encoding and decoding)
frames for the Hybi and Hixie protocols, respectively;

---

## Page 1959

Part 7. Advanced language tools
1 959
7.8 Projects
• wsmessage.mqh – WebSocketMessage and WebSocketMessageHixie classes that implement the
IWebSocketMessage interface, which formalizes the formation of messages from frames for the
Hybi and Hixie protocols, respectively;
• wsprotocol.mqh – WebSocketConnection, WebSocketConnectionHybi, WebSocketConnectionHixie
classes inherited from IWebSocketConnection; it is here that the coordinated management of the
formation of frames, messages, greetings, and disconnection according to the specification takes
place, for which the above interfaces are used;
• wsclient.mqh – ready-made implementation of a WebSocket client; a WebSocketClient template
class that supports the IWebSocketObserver interface (for event processing) and expects
WebSocketConnectionHybi or WebSocketConnectionHixie as a parameterized type;
• wstools.mqh – useful utilities in the WsTools namespace.
These header files will be automatically included in our future mqporj projects as dependencies from
#include directives.
WebSocket class diagram in MQL5
The low-level network interface IWebSocketTransport has the following methods.
interface IWebSocketTransport
{
   int write(const uchar &data[]); // write the array of bytes to the network
   int read(uchar &buffer[]);      // read data from network into byte array
   bool isConnected(void) const;   // check for connection
   bool isReadable(void) const;    // check for the ability to read from the network
   bool isWritable(void) const;    // check for the possibility of writing to the network
   int getHandle(void) const;      // system socket descriptor
   void close(void);               // close connection
};
It is not difficult to guess from the names of the methods which MQL5 API Socket functions will be
used to build them. But if necessary, those who wish can implement this interface by their own means,
for example, through a DLL.

---

## Page 1960

Part 7. Advanced language tools
1 960
7.8 Projects
The MqlWebSocketTransport class that implements this interface requires the protocol, hostname, and
port number to which the network connection is made when creating an instance. Additionally, you can
specify a timeout value.
Frame types are collected in the WS_FRAME_OPCODE enum.
enum WS_FRAME_OPCODE
{
   WS_DEFAULT = 0,
   WS_CONTINUATION_FRAME = 0x00,
   WS_TEXT_FRAME = 0x01,
   WS_BINARY_FRAME = 0x02,
   WS_CLOSE_FRAME = 0x08,
   WS_PING_FRAME = 0x09,
   WS_PONG_FRAME = 0x0A
};
The interface for working with frames contains both static and regular methods related to frame
instances. Static methods act as factories for creating frames of the required type by the transmitting
side (create) and incoming frames (decode).
class IWebSocketFrame
{
public:
   class StaticCreator
   {
   public:
      virtual IWebSocketFrame *decode(uchar &data[], IWebSocketFrame *head = NULL) = 0;
      virtual IWebSocketFrame *create(WS_FRAME_OPCODE type, const string data = NULL,
         const bool deflate = false) = 0;
      virtual IWebSocketFrame *create(WS_FRAME_OPCODE type, const uchar &data[],
         const bool deflate = false) = 0;
   };
   ...
The presence of factory methods in descendant classes is made mandatory due to the presence of a
template Creator and an instance of the getCreator method returning it (assuming return "singleton").

---

## Page 1961

Part 7. Advanced language tools
1 961 
7.8 Projects
protected:
   template<typename P>
   class Creator: public StaticCreator
   {
   public:
     // decode received binary data in IWebSocketFrame
     // (in case of continuation, previous frame in 'head')
      virtual IWebSocketFrame *decode(uchar &data[],
         IWebSocketFrame *head = NULL) override
      {
         return P::decode(data, head);
      }
      // create a frame of the desired type (text/closing/other) with optional text
      virtual IWebSocketFrame *create(WS_FRAME_OPCODE type, const string data = NULL,
         const bool deflate = false) override
      {
         return P::create(type, data, deflate);
      };
      // create a frame of the desired type (binary/text/closure/other) with data
      virtual IWebSocketFrame *create(WS_FRAME_OPCODE type, const uchar &data[],
         const bool deflate = false) override
      {
         return P::create(type, data, deflate);
      };
   };
public:
   // require a Creator instance
   virtual IWebSocketFrame::StaticCreator *getCreator() = 0;
   ...
The remaining methods of the interface provide all the necessary manipulations with data in frames
(encoding/decoding, receiving data and various flags).

---

## Page 1962

Part 7. Advanced language tools
1 962
7.8 Projects
   // encode the "clean" contents of the frame into data for transmission over the network
   virtual int encode(uchar &encoded[]) = 0;
   
   // get data as text
   virtual string getData() = 0;
   
   // get data as bytes, return size
   virtual int getData(uchar &buf[]) = 0;
   
   // return frame type (opcode)
   virtual WS_FRAME_OPCODE getType() = 0;
  
   // check if the frame is a control frame or with data:
   // control frames are processed inside classes
   virtual bool isControlFrame()
   {
      return (getType() >= WS_CLOSE_FRAME);
   }
   
   virtual bool isReady() { return true; }
   virtual bool isFinal() { return true; }
   virtual bool isMasked() { return false; }
   virtual bool isCompressed() { return false; }
};
The IWebSocketMessage interface contains methods for performing similar actions but at the message
level.

---

## Page 1963

Part 7. Advanced language tools
1 963
7.8 Projects
class IWebSocketMessage
{
public:
   // get an array of frames that make up this message
   virtual void getFrames(IWebSocketFrame *&frames[]) = 0;
   
   // set text as message content
   virtual bool setString(const string &data) = 0;
  
   // return message content as text
   virtual string getString() = 0;
  
   // set binary data as message content
   virtual bool setData(const uchar &data[]) = 0;
   
   // return the contents of the message in "raw" binary form
   virtual bool getData(uchar &data[]) = 0;
  
   // sign of completeness of the message (all frames received)
   virtual bool isFinalised() = 0;
  
   // add a frame to the message
   virtual bool takeFrame(IWebSocketFrame *frame) = 0;
};
Taking into account the interfaces of frames and messages, a common interface for WebSocket
connections IWebSocketConnection is defined.

---

## Page 1964

Part 7. Advanced language tools
1 964
7.8 Projects
interface IWebSocketConnection
{
   // open a connection with the specified URL and its parts,
   // and optional custom headers
   bool handshake(const string url, const string host, const string origin,
      const string custom = NULL);
   
   // low-level read frames from the server
   int readFrame(IWebSocketFrame *&frames[]);
   
   // low-level send frame (e.g. close or ping)
   bool sendFrame(IWebSocketFrame *frame);
   
   // low-level message sending
   bool sendMessage(IWebSocketMessage *msg);
   
   // custom check for new messages (event generation)
   int checkMessages();
   
   // custom text submission
   bool sendString(const string msg);
   
   // custom posting of binary data
   bool sendData(const uchar &data[]);
   
   // close the connection
   bool disconnect(void);
};
Notifications about disconnection and new messages are received via the IWebSocketObserver
interface methods.
interface IWebSocketObserver
{
  void onConnected();
  void onDisconnect();
  void onMessage(IWebSocketMessage *msg);
};
In particular, the WebSocketClient class was made a successor of this interface and by default simply
outputs information to the log. The class constructor expects an address to connect to the protocol ws
or wss.

---

## Page 1965

Part 7. Advanced language tools
1 965
7.8 Projects
template<typename T>
class WebSocketClient: public IWebSocketObserver
{
protected:
   IWebSocketMessage *messages[];
   
   string scheme;
   string host;
   string port;
   string origin;
   string url;
   int timeOut;
   ...
public:
   WebSocketClient(const string address)
   {
      string parts[];
      URL::parse(address, parts);
   
      url = address;
      timeOut = 5000;
  
      scheme = parts[URL_SCHEME];
      if(scheme != "ws" && scheme != "wss")
      {
        Print("WebSocket invalid url scheme: ", scheme);
        scheme = "ws";
      }
  
      host = parts[URL_HOST];
      port = parts[URL_PORT];
  
      origin = (scheme == "wss" ? "https://" : "http://") + host;
   }
   ...
  
   void onDisconnect() override
   {
      Print(" > Disconnected ", url);
   }
  
   void onConnected() override
   {
      Print(" > Connected ", url);
   }
  
   void onMessage(IWebSocketMessage *msg) override
   {
      // NB: message can be binary, print it just for notification
      Print(" > Message ", url, " " , msg.getString());
      WsTools::push(messages, msg);

---

## Page 1966

Part 7. Advanced language tools
1 966
7.8 Projects
   }
   ...
};
The WebSocketClient class collects all message objects into an array and takes care of deleting them if
the MQL program doesn't do it.
The connection is established in the open method.
template<typename T>
class WebSocketClient: public IWebSocketObserver
{
protected:
   IWebSocketTransport *socket;
   IWebSocketConnection *connection;
   ...
public:
   ...
   bool open(const string custom_headers = NULL)
   {
      uint _port = (uint)StringToInteger(port);
      if(_port == 0)
      {
         if(scheme == "ws") _port = 80;
         else _port = 443;
      }
  
      socket = MqlWebSocketTransport::create(scheme, host, _port, timeOut);
      if(!socket || !socket.isConnected())
      {
         return false;
      }
  
      connection = new T(&this, socket);
      return connection.handshake(url, host, origin, custom_headers);
   }
   ...
The most convenient ways to send data are provided by the overloaded send methods for text and
binary data.
   bool send(const string str)
   {
      return connection ? connection.sendString(str) : false;
   }
    
   bool send(const uchar &data[])
   {
      return connection ? connection.sendData(data) : false;
   }
To check for new incoming messages, you can call the checkMessages method. Depending on its
blocking parameter, the method will either wait for a message in a loop until the timeout or return

---

## Page 1967

Part 7. Advanced language tools
1 967
7.8 Projects
immediately if there are no messages. Messages will go to the IWebSocketObserver::onMessage
handler.
   void checkMessages(const bool blocking = true)
   {
      if(connection == NULL) return;
      
      uint stop = GetTickCount() + (blocking ? timeOut : 1);
      while(ArraySize(messages) == 0 && GetTickCount() < stop && isConnected())
      {
         // all frames are collected into the appropriate messages, and they become
         // available through event notifications IWebSocketObserver::onMessage,
         // however, control frames have already been internally processed and removed by now
         if(!connection.checkMessages()) // while no messages, let's make micro-pause
         {
            Sleep(100);
         }
      }
   }
An alternative way to receive messages is implemented in the readMessage method: it returns a pointer
to the message to the calling code (in other words, the application handler onMessage is not required).
After that, the MQL program is responsible for releasing the object.
   IWebSocketMessage *readMessage(const bool blocking = true)
   {
      if(ArraySize(messages) == 0) checkMessages(blocking);
      
      if(ArraySize(messages) > 0)
      {
         IWebSocketMessage *top = messages[0];
         ArrayRemove(messages, 0, 1);
         return top;
      }
      return NULL;
   }
The class also allows you to change the timeout, check the connection, and close it.

---

## Page 1968

Part 7. Advanced language tools
1 968
7.8 Projects
   void setTimeOut(const int ms)
   {
      timeOut = fabs(ms);
   }
   
   bool isConnected() const
   {
      return socket && socket.isConnected();
   }
   
   void close()
   {
      if(isConnected())
      {
         if(connection)
         {
            connection.disconnect(); // this will close socket after server acknowledge
            delete connection;
            connection = NULL;
         }
         if(socket)
         {
            delete socket;
            socket = NULL;
         }
      }
   }
};
The library of the considered classes allows you to create client applications for echo and chat
services.
7.8.7 Client programs for echo and chat services in MQL5
Let's write a simple script to connect to the echo service
MQL5/Experts/MQL5Book/p7/wsEcho/wsecho.mq5 (note that this is a script, but we placed it inside the
folder MQL5/Experts/MQL5Book/p7/, making it a single container for web-related MQL programs, since
all subsequent examples will be Experts Advisors). Since in this chapter, we are considering the creation
of software complexes within projects, we will design the script as part of an mqproj project, which will
also include the server component.
The input parameters of the script allow you to specify the address of the service and the text of the
message. The default is an unsecured connection. If you are going to launch the server wsecho.j s with
TLS support, you need to change the protocol to the secure wss. Keep in mind that establishing a
secure connection takes longer (by a couple of seconds) than usual.

---

## Page 1969

Part 7. Advanced language tools
1 969
7.8 Projects
input string Server = "ws://localhost:9000/";
input string Message = "My outbound message";
   
#include <MQL5Book/AutoPtr.mqh>
#include <MQL5Book/ws/wsclient.mqh>
In the OnStart function, we create an instance of the WebSocket client (wss) for the given address and
call the open method. In case of a successful connection, we wait for a welcome message from the
service by calling wss.readMessage in blocking mode (wait up to 5 seconds, by default). We use an
autopointer on the resulting object so as not to call delete manually at the end.
void OnStart()
{
   Print("\n");
   WebSocketClient<Hybi> wss(Server);
   Print("Opening...");
   if(wss.open())
   {
      Print("Waiting for welcome message (if any)");
      AutoPtr<IWebSocketMessage> welcome(wss.readMessage());
      ...
The WebSocketClient class contains event handler stubs, including the simple method onMessage,
which will print the greeting to the log.
Then we send our message and again wait for a response from the server. The echo message will also
be logged.
      Print("Sending message...");
      wss.send(Message);
      Print("Receiving echo...");
      AutoPtr<IWebSocketMessage> echo(wss.readMessage());
   }
   ...
Finally, we close the connection.
   if(wss.isConnected())
   {
      Print("Closing...");
      wss.close();
   }
}
Based on the script file, let's create a project file (wsecho.mqproj ). We fill in the project properties with
the version number (1 .0), copyright, and description. Let's add echo service server files to the Settings
and Files branch (this will at least remind the developer that there is a test server). After compilation,
dependencies (header files) will appear in the hierarchy.
Everything should look like in the screenshot below.

---

## Page 1970

Part 7. Advanced language tools
1 970
7.8 Projects
Echo service project, client script and server
If the script was located inside the folder Shared Proj ects, for example, in MQL5/Shared
Proj ects/MQL5Book/wsEcho/, then after successful compilation, its ex5 file would be automatically
moved to the folder MQL5/Scripts/Shared Proj ects/MQL5Book/wsEcho/, and the corresponding entry
would be displayed in the compilation log. This is the standard behavior for compiling any MQL
programs in shared projects.
In all examples of this chapter, do not forget to start the server before testing the MQL script. In this
case, run the command: node.exe wsecho.j s while in the web folder.
Next, let's run the script wsecho.ex5. The log will show the actions that are taking place, as well as the
message notifications.

---

## Page 1971

Part 7. Advanced language tools
1 971 
7.8 Projects
Opening...
Connecting to localhost:9000
Buffer: 'HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: mIpas63g5xGMqJcKtreHKpSbY1w=
'
Headers: 
                               [,0]                           [,1]
[0,] "upgrade"                      "websocket"                   
[1,] "connection"                   "Upgrade"                     
[2,] "sec-websocket-accept"         "mIpas63g5xGMqJcKtreHKpSbY1w="
 > Connected ws://localhost:9000/
Waiting for welcome message (if any)
 > Message ws://localhost:9000/ server#Hello, user1
Sending message...
Receiving echo...
 > Message ws://localhost:9000/ user1#My outbound message
Closing...
Close requested
Waiting...
SocketRead failed: 5273 Available: 1
 > Disconnected ws://localhost:9000/
Server close ack
The above HTTP headers are the server's response during the handshake process. If we look into the
console window where the server is running, we will find the HTTP headers received by the server from
our client.
Echo service server log
Also, the user's connection, message, and disconnection are indicated here.
Let's do a similar job for the chat service: create a WebSocket client in MQL5, a project for it, and test
it. This time the type of the client program will be an Expert Advisor because the chat needs support for
interactive events from the keyboard on the chart. The Expert Advisor is attached to the book in a
folder MQL5/MQL5Book/p7/wsChat/wschat.mq5.
To demonstrate the technology of receiving events in handler methods, let's define our own class
MyWebSocket, derived from WebSocketClient.

---

## Page 1972

Part 7. Advanced language tools
1 972
7.8 Projects
class MyWebSocket: public WebSocketClient<Hybi>
{
public:
   MyWebSocket(const string address, const bool compress = false):
      WebSocketClient(address, compress) { }
   
   /* void onConnected() override { } */
   
   void onDisconnect() override
   {
      // we can do something else and call (or not call) the legacy code
      WebSocketClient<Hybi>::onDisconnect();
   }
   
   void onMessage(IWebSocketMessage *msg) override
   {
     // TODO: we could truncate copies of our own messages,
     // but they are left for debugging
      Alert(msg.getString());
      delete msg;
   }
};
When a message is received, we will display it not in the log, but as an alert, after which the object
should be deleted.
In the global context, we describe the object of our wss class and the message string where the user
input from the keyboard will be accumulated.
MyWebSocket wss(Server);
string message = "";
The OnInit function contains the necessary preparation, in particular, starts a timer and opens a
connection.
int OnInit()
{
  ChartSetInteger(0, CHART_QUICK_NAVIGATION, false);
  EventSetTimer(1);
  wss.setTimeOut(1000);
  Print("Opening...");
  return wss.open() ? INIT_SUCCEEDED : INIT_FAILED;
}
The timer is needed to check for new messages from other users.
void OnTimer()
{
   wss.checkMessages(false); // use a non-blocking check in the timer
}
In the OnChartEvent handler, we respond to keystrokes: all alphanumeric keys are translated into
characters and attached to the message string. If necessary, you can press Backspace to remove the

---

## Page 1973

Part 7. Advanced language tools
1 973
7.8 Projects
last character. All typed text is updated in the chart comment. When the message is complete, press
Enter to send it to the server.
void OnChartEvent(const int id, const long &lparam, const double &dparam,
   const string &sparam)
{
   if(id == CHARTEVENT_KEYDOWN)
   {
      if(lparam == VK_RETURN)
      {
         const static string longmessage = ...
         if(message == "long") wss.send(longmessage);
         else if(message == "bye") wss.close();
         else wss.send(message);
         message = "";
      }
      else if(lparam == VK_BACK)
      {
         StringSetLength(message, StringLen(message) - 1);
      }
      else
      {
         ResetLastError();
         const short c = TranslateKey((int)lparam);
         if(_LastError == 0)
         {
            message += ShortToString(c);
         }
      }
      Comment(message);
   }
}
If we enter the text "long", the program will send a specially prepared rather long text. If the message
text is "bye", the program closes the connection. Also, the connection will be closed when the program
exits.
void OnDeinit(const int)
{
   if(wss.isConnected())
   {
      Print("Closing...");
      wss.close();
   }
}
Let's create a project for the Expert Advisor (file wschat.mqproj ), fill in its properties, and add the
backend to the branch Settings and Files. This time we will show how the project file looks from the
inside. In the mqproj file, the Dependencies branch is stored in the "files" property, and the Settings
and Files branch is in the "tester" property.

---

## Page 1974

Part 7. Advanced language tools
1 974
7.8 Projects
{
  "platform"    :"mt5",
  "program_type":"expert",
  "copyright"   :"Copyright 2022, MetaQuotes Ltd.",
  "version"     :"1.0",
  "description" :"WebSocket-client for chat-service.\r\nType and send text messages for all connected users.\r\nShow alerts with messages from others.",
  "optimize"    :"1",
  "fpzerocheck" :"1",
  "tester_no_cache":"0",
  "tester_everytick_calculate":"0",
  "unicode_character_set":"0",
  "static_libraries":"0",
  "files":
  [
    {
      "path":"wschat.mq5",
      "compile":true,
      "relative_to_project":true
    },
    {
      "path":"MQL5\\Include\\MQL5Book\\ws\\wsclient.mqh",
      "compile":false,
      "relative_to_project":false
    },
    {
      "path":"MQL5\\Include\\MQL5Book\\URL.mqh",
      "compile":false,
      "relative_to_project":false
    },
    {
      "path":"MQL5\\Include\\MQL5Book\\ws\\wsframe.mqh",
      "compile":false,
      "relative_to_project":false
    },
    {
      "path":"MQL5\\Include\\MQL5Book\\ws\\wstools.mqh",
      "compile":false,
      "relative_to_project":false
    },
    {
      "path":"MQL5\\Include\\MQL5Book\\ws\\wsinterfaces.mqh",
      "compile":false,
      "relative_to_project":false
    },
    {
      "path":"MQL5\\Include\\MQL5Book\\ws\\wsmessage.mqh",
      "compile":false,
      "relative_to_project":false
    },
    {
      "path":"MQL5\\Include\\MQL5Book\\ws\\wstransport.mqh",
      "compile":false,
      "relative_to_project":false
    },

---

## Page 1975

Part 7. Advanced language tools
1 975
7.8 Projects
    {
      "path":"MQL5\\Include\\MQL5Book\\ws\\wsprotocol.mqh",
      "compile":false,
      "relative_to_project":false
    },
    {
      "path":"MQL5\\Include\\VirtualKeys.mqh",
      "compile":false,
      "relative_to_project":false
    }
  ],
  "tester":
  [
    {
      "type":"file",
      "path":"..\\Web\\MQL5Book.crt",
      "relative_to_project":true
    },
    {
      "type":"file",
      "path":"..\\Web\\MQL5Book.key",
      "relative_to_project":true
    },
    {
      "type":"file",
      "path":"..\\Web\\wschat.htm",
      "relative_to_project":true
    },
    {
      "type":"file",
      "path":"..\\Web\\wschat.js",
      "relative_to_project":true
    },
    {
      "type":"file",
      "path":"..\\Web\\wschat_client.js",
      "relative_to_project":true
    }
  ]
}
If the Expert Advisor were inside the Shared Proj ects folder, for example, in MQL5/Shared
Proj ects/MQL5Book/wsChat/, after successful compilation, its ex5 file would be automatically moved to
the folder MQL5/Experts/Shared Proj ects/MQL5Book/wsChat/.
Starting the server node.exe wschat.j s. Now you can run a couple of copies of the Expert Advisor on
different charts. Basically, the service involves "communication" between different terminals and even
different computers, but you can also test it from one terminal.
Here is an example of communication between the EURUSD and GBPUSD charts.

---

## Page 1976

Part 7. Advanced language tools
1 976
7.8 Projects
(EURUSD,H1)
(EURUSD,H1)Opening...
(EURUSD,H1)Connecting to localhost:9000
(EURUSD,H1)Buffer: 'HTTP/1.1 101 Switching Protocols
(EURUSD,H1)Upgrade: websocket
(EURUSD,H1)Connection: Upgrade
(EURUSD,H1)Sec-WebSocket-Accept: Dg+aQdCBwNExE5mEQsfk5w9J+uE=
(EURUSD,H1)
(EURUSD,H1)'
(EURUSD,H1)Headers: 
(EURUSD,H1)                               [,0]                           [,1]
(EURUSD,H1)[0,] "upgrade"                      "websocket"                   
(EURUSD,H1)[1,] "connection"                   "Upgrade"                     
(EURUSD,H1)[2,] "sec-websocket-accept"         "Dg+aQdCBwNExE5mEQsfk5w9J+uE="
(EURUSD,H1) > Connected ws://localhost:9000/
(EURUSD,H1)Alert: server#Hello, user1
(GBPUSD,H1)
(GBPUSD,H1)Opening...
(GBPUSD,H1)Connecting to localhost:9000
(GBPUSD,H1)Buffer: 'HTTP/1.1 101 Switching Protocols
(GBPUSD,H1)Upgrade: websocket
(GBPUSD,H1)Connection: Upgrade
(GBPUSD,H1)Sec-WebSocket-Accept: NZENnc8p05T4amvngeop/e/+gFw=
(GBPUSD,H1)
(GBPUSD,H1)'
(GBPUSD,H1)Headers: 
(GBPUSD,H1)                               [,0]                           [,1]
(GBPUSD,H1)[0,] "upgrade"                      "websocket"                   
(GBPUSD,H1)[1,] "connection"                   "Upgrade"                     
(GBPUSD,H1)[2,] "sec-websocket-accept"         "NZENnc8p05T4amvngeop/e/+gFw="
(GBPUSD,H1) > Connected ws://localhost:9000/
(GBPUSD,H1)Alert: server#Hello, user2
(EURUSD,H1)Alert: user1#I'm typing this on EURUSD chart
(GBPUSD,H1)Alert: user1#I'm typing this on EURUSD chart
(GBPUSD,H1)Alert: user2#Got it on GBPUSD chart!
(EURUSD,H1)Alert: user2#Got it on GBPUSD chart!
Since our messages are sent to everyone, including the sender, they are duplicated in the log, but on
different charts.
Communication is visible on the server side as well.

---

## Page 1977

Part 7. Advanced language tools
1 977
7.8 Projects
Chat service server log
Now we have all the technical components for organizing the trading signals service.
7.8.8 Trading signal service and test web page
The trading signal service is technically identical to the chat service, however, its users (or rather client
connections) must perform one of two roles:
• Message provider
• Message consumer
In addition, the information should not be available to everyone but work according to some
subscription scheme.
To ensure this, when connecting to the service, users will be required to provide certain identifying
information that differs depending on the role.
The provider must specify a public signal identifier (PUB_ID) that is unique among all signals. Basically,
the same person could potentially generate more than one signal and should therefore be able to obtain
multiple identifiers. In this sense, we will not complicate the service by introducing separate provider
identifiers (as a specific person) and identifiers of its signals. Instead, only signal identifiers will be
supported. For a real signal service, this issue needs to be worked out, along with authorization, which
we left outside of this book.
The identifier will be required in order to advertise it or simply pass it on to persons interested in
subscribing to this signal. But "everyone you meet" should not be able to access the signal knowing
only the public identifier. In the simplest case, this would be acceptable for open account monitoring,
but we will demonstrate the option of restricting access specifically in the context of signals.
For this purpose, the provider must provide the server with a secret key (PUB_KEY) known only to them
but not to the public. This key will be required to generate a specific subscriber's access key.
The consumer (subscriber) must also have a unique identifier (SUB_ID, and here we will also do without
authorization). To subscribe to the desired signal, the user must tell the signal provider the identifier (in
practice, it is understood that at the same stage, it is necessary to confirm the payment, and usually
this is all automated by the server). The provider forms a snapshot consisting of the provider's
identifier, the subscriber's identifier, and the secret key. In our service, this will be done by calculating

---

## Page 1978

Part 7. Advanced language tools
1 978
7.8 Projects
the SHA256 hash from the PUB_ID:PUB_KEY:SUB_ID string, after which the resulting bytes are
converted to a hexadecimal format string. This will be the access key (SUB_KEY or ACCESS_KEY) to
the signal of a particular provider for a particular subscriber. The provider (and in real systems, the
server itself automatically) forwards this key to the subscriber.
Thus, when connecting to the service, the subscriber will have to specify the subscriber identifier
(SUB_ID), the identifier of the desired signal (PUB_ID), and the access key (SUB_KEY). Because the
server knows the provider's secret key, it can recalculate the access key for the given combination of
PUB_ID and SUB_ID, and compare it with the provided SUB_KEY. A match means the normal
messaging process continues. The difference will result in an error message and disconnecting the
pseudo-subscriber from the service.
It is important to note that in our demo, for the sake of simplicity, there is no normal registration of
users and signals, and therefore the choice of identifiers is arbitrary. It is only important for us to keep
track of the uniqueness of identifiers in order to know to whom and from whom to send information
online. So, our service does not guarantee that the identifier, for example, "Super Trend" belongs to
the same user yesterday, today, and tomorrow. Reservation of names is made according to the
principle that the early bird catches the worm. As long as a provider is continuously connected under
the given identifier, the signal is delivered. If the provider disconnects, then the identifier becomes
available for selection in any next connection.
The only identifier that will always be busy is "Server": the server uses it to send out its connection
status messages.
To generate access keys in the server folder, there is a simple JavaScript access.j s. When you run it on
the command line, you need to pass as the only parameter a string of the above type
PUB_ID:PUB_KEY:SUB_ID (identifiers and the secret key between them, connected by the ':' symbol)
If the parameter is not specified, the script generates an access key for some demo identifiers
(PUB_ID_001 , SUB_ID_1 00) and a secret (PUB_KEY_FFF).
// JavaScript
const args = process.argv.slice(2);
const input = args.length > 0 ? args[0] : 'PUB_ID_001:PUB_KEY_FFF:SUB_ID_100';
console.log('Hashing "', input, '"');
const crypto = require('crypto');
console.log(crypto.createHash('sha256').update(input).digest('hex'));
Running the script with the command:
node access.js PUB_ID_001:PUB_KEY_FFF:SUB_ID_100
we get this result:
fd3f7a105eae8c2d9afce0a7a4e11bf267a40f04b7c216dd01cf78c7165a2a5a
By the way, you can check and repeat this algorithm in pure MQL5 using the CryptEncode function.
Having analyzed the conceptual part, let's proceed to practical implementation.
The server script of the signaling service will be placed in the file
MQL5/Experts/MQL5Book/p7/Web/wspubsub.j s. Setting up servers in it is the same as what we did
earlier. However, in addition, you will need to connect the same "crypto" module that was used in
access.j s. The home page will be called wspubsub.htm.

---

## Page 1979

Part 7. Advanced language tools
1 979
7.8 Projects
// JavaScript
const crypto = require('crypto');
...
http1.createServer(options, function (req, res)
{
   ...
   if(req.url == '/')
   {
      req.url = "wspubsub.htm";
   }
   ...
});
Instead of one map of connected clients, we will define two maps, separately for signal providers and
consumers.
// JavaScript
const publishers = new Map();
const subscribers = new Map();
In both maps, the key is the provider ID, but the first one stores the objects of the providers, and the
second one stores the objects of subscribers subscribed to each provider (arrays of objects).
To transfer identifiers and keys during the handshake, we will use a special header allowed by the
WebSockets specification, namely Sec-Websocket-Protocol. Let's agree that identifiers and keys will
be glued together with the symbol '-': in the case of a provider, a string like X-MQL5-publisher-PUB_ID-
PUB_KEY is expected, and in the case of a subscriber, we expect X-MQL5-subscriber-SUB_ID-PUB_ID-
SUB_KEY.
Any attempts to connect to our service without the Sec-Websocket-Protocol: X-MQL5-... header will be
stopped by immediate closure.
In the new client object (in the "connection" event handler parameter onConnect(client)) this title is
easy to extract from the client.protocol property.
Let's show the procedure for registering and sending the signal provider's messages in a simplified
form, without error handling (the full code is attached). It is important to note that the message text is
generated in JSON format (which we will discuss in more detail in the next section). In particular, the
sender of the message is passed in the "origin" property (moreover, when the message is sent by the
service itself, this field contains the string "Server"), and the application data from the provider is
placed in the "msg" property, and this may not be just text, but also nested structure of any content.

---

## Page 1980

Part 7. Advanced language tools
1 980
7.8 Projects
// JavaScript
const wsServer = new WebSocket.Server({ server });
wsServer.on('connection', function onConnect(client)
{
   console.log('New user:', ++count, client.protocol);
   if(client.protocol.startsWith('X-MQL5-publisher'))
   {
      const parts = client.protocol.split('-');
      client.id = parts[3];
      client.key = parts[4];
      publishers.set(client.id, client);
      client.send('{"origin":"Server", "msg":"Hello, publisher ' + client.id + '"}');
      client.on('message', function(message)
      {
         console.log('%s : %s', client.id, message);
         
         if(subscribers.get(client.id))
            subscribers.get(client.id).forEach(function(elem)
         {
            elem.send('{"origin":"publisher ' + client.id + '", "msg":'
               + message + '}');
         });
      });
      client.on('close', function()
      {
         console.log('Publisher disconnected:', client.id);
         if(subscribers.get(client.id))
            subscribers.get(client.id).forEach(function(elem)
         {
            elem.close();
         });
         publishers.delete(client.id);
      });
   }
   ...
Half of the algorithm for subscribers is similar, but here we have the calculation of the access key and
its comparison with what the connecting client transmitted, as an addition.

---

## Page 1981

Part 7. Advanced language tools
1 981 
7.8 Projects
// JavaScript
   else if(client.protocol.startsWith('X-MQL5-subscriber'))
   {
      const parts = client.protocol.split('-');
      client.id = parts[3];
      client.pub_id = parts[4];
      client.access = parts[5];
      const id = client.pub_id;
      var p = publishers.get(id);
      if(p)
      {
         const check = crypto.createHash('sha256').update(id + ':' + p.key + ':'
            + client.id).digest('hex');
         if(check != client.access)
         {
            console.log(`Bad credentials: '${client.access}' vs '${check}'`);
            client.send('{"origin":"Server", "msg":"Bad credentials, subscriber '
               + client.id + '"}');
            client.close();
            return;
         }
   
         var list = subscribers.get(id);
         if(list == undefined)
         {
            list = [];
         }
         list.push(client);
         subscribers.set(id, list);
         client.send('{"origin":"Server", "msg":"Hello, subscriber '
            + client.id + '"}');
         p.send('{"origin":"Server", "msg":"New subscriber ' + client.id + '"}');
      }
      
      client.on('close', function()
      {
         console.log('Subscriber disconnected:', client.id);
         const list = subscribers.get(client.pub_id);
         if(list)
         {
            if(list.length > 1)
            {
               const filtered = list.filter(function(el) { return el !== client; });
               subscribers.set(client.pub_id, filtered);
            }
            else
            {
               subscribers.delete(client.pub_id);
            }
         }
      });

---

## Page 1982

Part 7. Advanced language tools
1 982
7.8 Projects
   }
The user interface on the client page wspubsub.htm simply invites you to follow a link to one of the two
pages with forms for suppliers (wspublisher.htm + wspublisher_ client.j s) or subscribers
(wssubscriber.htm + wssubscriber_ client.j s).
Web pages of signal service test clients
Their implementation inherits the features of the previously considered JavaScript clients, but with
respect to the customization of the Sec-Websocket-Protocol: X-MQL5- header and one more nuance.
Until now, we have exchanged simple text messages. But for a signaling service, you will need to
transfer a lot of structured information, and JSON is better suited for this. Therefore, clients can parse
JSON, although they do not use it for its intended purpose, because even if a command to buy or sell a
specific ticker with a given amount is found in JSON, the browser does not know how to do this.
We will need to add JSON support to our signal service client in MQL5. Meanwhile, you can run on the
server wspubsub.j s and test the selective connection of signal providers and consumers in accordance
with the details specified by them. We suggest you do it yourself, for your own benefit.
7.8.9 Signal service client program in MQL5
So, according to our decision, the text in the service messages will be in JSON format.
In the most common version, JSON is a text description of an object, similar to how it is done for
structures in MQL5. The object is enclosed in curly brackets, inside which its properties are written
separated by commas: each property has an identifier in quotes, followed by a colon and the value of
the property. Here properties of several primitive types are supported: strings, integers and real
numbers, booleans true/false, and empty value null. In addition, the property value can, in turn, be an

---

## Page 1983

Part 7. Advanced language tools
1 983
7.8 Projects
object or an array. Arrays are described using square brackets, within which the elements are
separated by commas. For example,
{
   "string": "this is a text",
   "number": 0.1,
   "integer": 789735095,
   "enabled": true,
   "subobject" :
   {
      "option": null
   },
   "array":
   [
      1, 2, 3, 5, 8
   ]
}
Basically, the array at the top level is also valid JSON. For example,
[
   {
      "command": "buy",
      "volume": 0.1,
      "symbol": "EURUSD",
      "price": 1.0
   },
   {
      "command": "sell",
      "volume": 0.01,
      "symbol": "GBPUSD",
      "price": 1.5
   }
]
To reduce traffic in application protocols using JSON, it is customary to abbreviate field names to
several letters (often to one).
Property names and string values are enclosed in double-quotes. If you want to specify a quote within a
string, it must be escaped with a backslash.
The use of JSON makes the protocol versatile and extensible. For example, for the service being
designed (trading signals and, in a more general case, account state copying), the following message
structure can be assumed:

---

## Page 1984

Part 7. Advanced language tools
1 984
7.8 Projects
{
  "origin": "publisher_id",    // message sender ("Server" in technical message)
  "msg" :                      // message (text or JSON) as received from the sender
   {
     "trade" :                 // current trading commands (if there is a signal)
      {
        "operation": ...,      // buy/sell/close
         "symbol": "ticker",
         "volume": 0.1,
        ... // other signal parameters
      },
     "account":                // account status
      {
        "positions":           // positions
         {
           "n": 10,            // number of open positions
           [ { ... },{ ... } ] // array of properties of open positions
         },
        "pending_orders":      // pending orders
         {
            "n": ...
            [ { ... } ]
         }
         "drawdown": 2.56,
         "margin_level": 12345,
        ... // other status parameters
      },
     "hardware":               // remote control of the "health" of the PC
      {
         "memory": ...,
         "ping_to_broker": ...
      }
   }
}
Some of these features may or may not support specific implementations of client programs
(everything that they do not "understand", they will simply ignore). In addition, subject to the condition
that there are no conflicts in the names of properties at the same level, each information provider can
add its own specific data to JSON. The messaging service will simply forward this information. Of
course, the program on the receiving side must be able to interpret these specific data.
The book comes with a JSON parser called ToyJson ("toy" JSON, file toyj son.mqh) which is small and
inefficient and does not support the full capabilities of the format specification (for example, in terms of
processing of escape sequences). It was written specifically for this demo service, adjusted for the
expected, not very complex, structure of information about trading signals. We will not describe it in
detail here, and the principles of its use will become clear from the source code of the MQL client of the
signal service.
For your projects and for the further development of this project, you can choose other JSON parsers
available in the codebase on the mql5.com site.

---

## Page 1985

Part 7. Advanced language tools
1 985
7.8 Projects
One element (container or property) per ToyJson is described by the JsValue class object. There are
several overloads of the method put(key, value) defined, that can be used for the addition of named
internal properties as in a JSON object or put(value), to add a value as in a JSON array. Also, this
object can represent a single value of a primitive type. To read the properties of a JSON object, you
can apply to JsValue a notation of the operator [] followed by the required property name in
parentheses. Obviously, integer indexes are supported for accessing inside a JSON array.
Having formed the required configuration of related objects JsValue, you can serialize it into JSON text
using the stringify(string&buffer) method.
The second class in toyj son.mqh – JsParser – allows you to perform the reverse operation: turn the text
with the JSON description into a hierarchical structure of JsValue objects.
Taking into account the classes for working with JSON, let's start writing an Expert Advisor
MQL5/Experts/MQL5Book/p7/wsTradeCopier/wstradecopier.mq5, which will be able to perform both
roles in the transaction copy service: a provider of information about trades made on the account or a
recipient of this information from the service to reproduce these trades.
The volume and content of the information sent is, from a political point of view, at the discretion of the
provider and may differ significantly depending on the scenario (purpose) of using the service. In
particular, it is possible to copy only ongoing transactions or the entire account balance along with
pending orders and protective levels. In our example, we will only indicate the technical implementation
of information transfer, and then you can choose a specific set of objects and properties at your
discretion.
In the code, we will describe 3 structures which are inherited from built-in structures and which provide
information "packing" in JSON:
• MqlTradeRequestWeb – MqlTradeRequest
• MqlTradeResultWeb – MqlTradeResult
• DealMonitorWeb – DealMonitor*
The last structure in the list, strictly speaking, is not built-in, but is defined by us in the file
DealMonitor.mqh, yet it is filled on the standard set of deal properties.
The constructor of each of the derived structures populates the fields based on the transmitted primary
source (trade request, its result, or deal). Each structure implements the asJsValue method, which
returns a pointer to the JsValue object that reflects all the properties of the structure: they are added
to the JSON object using the JsValue::put method. For example, here is how it is done in the case of
MqlTradeRequest:

---

## Page 1986

Part 7. Advanced language tools
1 986
7.8 Projects
struct MqlTradeRequestWeb: public MqlTradeRequest
{
   MqlTradeRequestWeb(const MqlTradeRequest &r)
   {
      ZeroMemory(this);
      action = r.action;
      magic = r.magic;
      order = r.order;
      symbol = r.symbol;
      volume = r.volume;
      price = r.price;
      stoplimit = r.stoplimit;
      sl = r.sl;
      tp = r.tp;
      type = r.type;
      type_filling = r.type_filling;
      type_time = r.type_time;
      expiration = r.expiration;
      comment = r.comment;
      position = r.position;
      position_by = r.position_by;
   }
   
   JsValue *asJsValue() const
   {
      JsValue *req = new JsValue();
      // main block: action, symbol, type
      req.put("a", VerboseJson ? EnumToString(action) : (string)action);
      if(StringLen(symbol) != 0) req.put("s", symbol);
      req.put("t", VerboseJson ? EnumToString(type) : (string)type);
      
      // volumes
      if(volume != 0) req.put("v", TU::StringOf(volume));
      req.put("f", VerboseJson ? EnumToString(type_filling) : (string)type_filling);
      
      // block with prices
      if(price != 0) req.put("p", TU::StringOf(price));
      if(stoplimit != 0) req.put("x", TU::StringOf(stoplimit));
      if(sl != 0) req.put("sl", TU::StringOf(sl));
      if(tp != 0) req.put("tp", TU::StringOf(tp));
   
      // block of pending orders
      if(TU::IsPendingType(type))
      {
         req.put("t", VerboseJson ? EnumToString(type_time) : (string)type_time);
         if(expiration != 0) req.put("d", TimeToString(expiration));
      }
   
      // modification block
      if(order != 0) req.put("o", order);
      if(position != 0) req.put("q", position);

---

## Page 1987

Part 7. Advanced language tools
1 987
7.8 Projects
      if(position_by != 0) req.put("b", position_by);
      
      // helper block
      if(magic != 0) req.put("m", magic);
      if(StringLen(comment)) req.put("c", comment);
   
      return req;
   }
};
We transfer all properties to JSON (this is suitable for the account monitoring service), but you can
leave only a limited set.
For properties that are enumerations, we have provided two ways to represent them in JSON: as an
integer and as a string name of an enumeration element. The choice of method is made using the input
parameter VerboseJson (ideally, it should be written in the structure code not directly but through a
constructor parameter).
input bool VerboseJson = false;
Passing only numbers would simplify coding because, on the receiving side, it is enough to cast them to
the desired enumeration type in order to perform "mirror" actions. However, numbers make it difficult
for a person to perceive information, and they may need to analyze the situation (message). Therefore,
it makes sense to support an option for the string representation, as being more "friendly", although it
requires additional operations in the receiving algorithm.
The input parameters also specify the server address, the application role, and connection details
separately for the provider and the subscriber.
enum TRADE_ROLE
{
   TRADE_PUBLISHER,  // Trade Publisher
   TRADE_SUBSCRIBER  // Trade Subscriber
};
   
input string Server = "ws://localhost:9000/";
input TRADE_ROLE Role = TRADE_PUBLISHER;
input bool VerboseJson = false;
input group "Publisher";
input string PublisherID = "PUB_ID_001";
input string PublisherPrivateKey = "PUB_KEY_FFF";
input string SymbolFilter = ""; // SymbolFilter (empty - current, '*' - any)
input ulong MagicFilter = 0;    // MagicFilter (0 - any)
input group "Subscriber";
input string SubscriberID = "SUB_ID_100";
input string SubscribeToPublisherID = "PUB_ID_001";
input string SubscriberAccessKey = "fd3f7a105eae8c2d9afce0a7a4e11bf267a40f04b7c216dd01cf78c7165a2a5a";
input string SymbolSubstitute = "EURUSD=GBPUSD"; // SymbolSubstitute (<from>=<to>,...)
input ulong SubscriberMagic = 0;
Parameters SymbolFilter and MagicFilter in the provider group allow you to limit the monitored trading
activity to a given symbol and magic number. An empty value in SymbolFilter means to control only the
current symbol of the chart, to intercept any trades, enter the symbol '*'. The signal provider will use

---

## Page 1988

Part 7. Advanced language tools
1 988
7.8 Projects
for this purpose the FilterMatched function, which accepts the symbol and magic number of the
transaction.
bool FilterMatched(const string s, const ulong m)
{
   if(MagicFilter != 0 && MagicFilter != m)
   {
      return false;
   }
   
   if(StringLen(SymbolFilter) == 0)
   {
      if(s != _Symbol)
      {
         return false;
      }
   }
   else if(SymbolFilter != s && SymbolFilter != "*")
   {
      return false;
   }
   
   return true;
}
The SymbolSubstitute parameter in the input group of the subscriber allows the substitution of the
symbol received in messages with another one, which will be used for copy trading. This feature is
useful if the names of tickers of the same financial instrument differ between brokers. But this
parameter also performs the function of a permissive filter for repeating signals: only the symbols
specified here will be traded. For example, to allow signal trading for the EURUSD symbol (even without
ticker substitution), you need to set the string "EURUSD=EURUSD" in the parameter. The symbol from
the signal messages is indicated to the left of the sign '=', and the symbol for trading is indicated to the
right.
The character substitution list is processed by the FillSubstitutes function during initialization and then
used to substitute and resolve the trade by the FindSubstitute function.

---

## Page 1989

Part 7. Advanced language tools
1 989
7.8 Projects
string Substitutes[][2];
   
void FillSubstitutes()
{
   string list[];
   const int n = StringSplit(SymbolSubstitute, ',', list);
   ArrayResize(Substitutes, n);
   for(int i = 0; i < n; ++i)
   {
      string pair[];
      if(StringSplit(list[i], '=', pair) == 2)
      {
         Substitutes[i][0] = pair[0];
         Substitutes[i][1] = pair[1];
      }
      else
      {
         Print("Wrong substitute: ", list[i]);
      }
   }
}
   
string FindSubstitute(const string s)
{
   for(int i = 0; i < ArrayRange(Substitutes, 0); ++i)
   {
      if(Substitutes[i][0] == s) return Substitutes[i][1];
   }
   return NULL;
}
To communicate with the service, we define a class derived from WebSocketClient. It is needed, first of
all, to start trading on a signal when a message arrives in the onMessage handler. We will return to this
issue a little later after we consider the formation and sending of signals on the provider side.
class MyWebSocket: public WebSocketClient<Hybi>
{
public:
   MyWebSocket(const string address): WebSocketClient(address) { }
   
   void onMessage(IWebSocketMessage *msg) override
   {
      ...
   }
};
   
MyWebSocket wss(Server);
Initialization in OnInit turns on the timer (for a periodic call wss.checkMessages(false)) and preparation
of custom headers with user details, depending on the selected role. Then we open the connection with
the wss.open(custom) call.

---

## Page 1990

Part 7. Advanced language tools
1 990
7.8 Projects
int OnInit()
{
   FillSubstitutes();
   EventSetTimer(1);
   wss.setTimeOut(1000);
   Print("Opening...");
   string custom;
   if(Role == TRADE_PUBLISHER)
   {
      custom = "Sec-Websocket-Protocol: X-MQL5-publisher-"
         + PublisherID + "-" + PublisherPrivateKey + "\r\n";
   }
   else
   {
      custom = "Sec-Websocket-Protocol: X-MQL5-subscriber-"
         + SubscriberID + "-" + SubscribeToPublisherID
         + "-" + SubscriberAccessKey + "\r\n";
   }
   return wss.open(custom) ? INIT_SUCCEEDED : INIT_FAILED;
}
The mechanism of copying, i.e., intercepting transactions and sending information about them to a web
service, is launched in the OnTradeTransaction handler. As we know, this is not the only way and it
would be possible to analyze the "snapshot" of the account state in OnTrade.
void OnTradeTransaction(const MqlTradeTransaction &transaction,
   const MqlTradeRequest &request,
   const MqlTradeResult &result)
{
   if(transaction.type == TRADE_TRANSACTION_REQUEST)
   {
      Print(TU::StringOf(request));
      Print(TU::StringOf(result));
      if(result.retcode == TRADE_RETCODE_PLACED           // successful action
         || result.retcode == TRADE_RETCODE_DONE
         || result.retcode == TRADE_RETCODE_DONE_PARTIAL)
      {
         if(FilterMatched(request.symbol, request.magic))
         {
            ... // see next block of code
         }
      }
   }
}
We track events about successfully completed trade requests that satisfy the conditions of the
specified filters. Next, the structures of the request, the result of the request, and the deal are turned
into JSON objects. All of them are placed in one common container msg under the names "req", "res",
and "deal", respectively. Recall that the container itself will be included in the web service message as
the "msg" property.

---

## Page 1991

Part 7. Advanced language tools
1 991 
7.8 Projects
           // container to attach to service message will be visible as "msg" property:
             // {"origin" : "this_publisher_id", "msg" : { our data is here }}
            JsValue msg;
            MqlTradeRequestWeb req(request);
            msg.put("req", req.asJsValue());
            
            MqlTradeResultWeb res(result);
            msg.put("res", res.asJsValue());
            
            if(result.deal != 0)
            {
               DealMonitorWeb deal(result.deal);
               msg.put("deal", deal.asJsValue());
            }
            ulong tickets[];
            Positions.select(tickets);
            JsValue pos;
            pos.put("n", ArraySize(tickets));
            msg.put("pos", &pos);
            string buffer;
            msg.stringify(buffer);
            
            Print(buffer);
            
            wss.send(buffer);
Once filled, the container is output as a string into buffer, printed to the log, and sent to the server.
We can add other information to this container: account status (drawdown, loading), the number and
properties of pending orders, and so on. So, just to demonstrate the possibilities for expanding the
content of messages, we have added the number of open positions above. To select positions according
to filters, we used the PositionFilter class object (PositionFilter.mqh):
PositionFilter Positions;
   
int OnInit()
{
   ...
   if(MagicFilter) Positions.let(POSITION_MAGIC, MagicFilter);
   if(SymbolFilter == "") Positions.let(POSITION_SYMBOL, _Symbol);
   else if(SymbolFilter != "*") Positions.let(POSITION_SYMBOL, SymbolFilter);
   ...
}
Basically, in order to increase reliability, it makes sense for the copiers to analyze the state of
positions, and not just intercept transactions.
This concludes the consideration of the part of the Expert Advisor that is involved in the role of the
signal provider.
As a subscriber, as we have already announced, the Expert Advisor receives messages in the
MyWebSocket::onMessage method. Here the incoming message is parsed with JsParser::j sonify, and the
container that was formed by the transmitting side is retrieved from the obj ["msg"] property.

---

## Page 1992

Part 7. Advanced language tools
1 992
7.8 Projects
class MyWebSocket: public WebSocketClient<Hybi>
{
public:
   void onMessage(IWebSocketMessage *msg) override
   {
      Alert(msg.getString());
      JsValue *obj = JsParser::jsonify(msg.getString());
      if(obj && obj["msg"])
      {
         obj["msg"].print();
         if(!RemoteTrade(obj["msg"])) { /* error processing */ }
         delete obj;
      }
      delete msg;
   }
};
The RemoteTrade function implements the signal analysis and trading operations. Here it is given with
abbreviations, without handling potential errors. The function provides support for both ways of
representing enumerations: as integer values or as string element names. The incoming JSON object is
"examined" for the necessary properties (commands and signal attributes) by applying the operator [],
including several times consecutively (to access nested JSON objects).

---

## Page 1993

Part 7. Advanced language tools
1 993
7.8 Projects
bool RemoteTrade(JsValue *obj)
{
   bool success = false;
   
   if(obj["req"]["a"] == TRADE_ACTION_DEAL
      || obj["req"]["a"] == "TRADE_ACTION_DEAL")
   {
      const string symbol = FindSubstitute(obj["req"]["s"].s);
      if(symbol == NULL)
      {
         Print("Suitable symbol not found for ", obj["req"]["s"].s);
         return false; // not found or forbidden
      }
      
      JsValue *pType = obj["req"]["t"];
      if(pType == ORDER_TYPE_BUY || pType == ORDER_TYPE_SELL
         || pType == "ORDER_TYPE_BUY" || pType == "ORDER_TYPE_SELL")
      {
         ENUM_ORDER_TYPE type;
         if(pType.detect() >= JS_STRING)
         {
            if(pType == "ORDER_TYPE_BUY") type = ORDER_TYPE_BUY;
            else type = ORDER_TYPE_SELL;
         }
         else
         {
            type = obj["req"]["t"].get<ENUM_ORDER_TYPE>();
         }
         
         MqlTradeRequestSync request;
         request.deviation = 10;
         request.magic = SubscriberMagic;
         request.type = type;
         
         const double lot = obj["req"]["v"].get<double>();
         JsValue *pDir = obj["deal"]["entry"];
         if(pDir == DEAL_ENTRY_IN || pDir == "DEAL_ENTRY_IN")
         {
            success = request._market(symbol, lot) && request.completed();
            Alert(StringFormat("Trade by subscription: market entry %s %s %s - %s",
               EnumToString(type), TU::StringOf(lot), symbol,
               success ? "Successful" : "Failed"));
         }
         else if(pDir == DEAL_ENTRY_OUT || pDir == "DEAL_ENTRY_OUT")
         {
            // closing action assumes the presence of a suitable position, look for it
            PositionFilter filter;
            int props[] = {POSITION_TICKET, POSITION_TYPE, POSITION_VOLUME};
            Tuple3<long,long,double> values[];
            filter.let(POSITION_SYMBOL, symbol).let(POSITION_MAGIC,
               SubscriberMagic).select(props, values);

---

## Page 1994

Part 7. Advanced language tools
1 994
7.8 Projects
            for(int i = 0; i < ArraySize(values); ++i)
            {
              // need a position that is opposite in direction to the deal
               if(!TU::IsSameType((ENUM_ORDER_TYPE)values[i]._2, type))
               {
                  // you need enough volume (exactly equal here!)
                  if(TU::Equal(values[i]._3, lot))
                  {
                     success = request.close(values[i]._1, lot) && request.completed();
                     Alert(StringFormat("Trade by subscription: market exit %s %s %s - %s",
                        EnumToString(type), TU::StringOf(lot), symbol,
                        success ? "Successful" : "Failed"));
                  }
               }
            }
            
            if(!success)
            {
               Print("No suitable position to close");
            }
         }
      }
   }
   return success;
}
This implementation does not analyze the transaction price, possible restrictions on the lot, stop levels,
and other moments. We simply repeat the trade at the current local price. Also, when closing a
position, a check is made for exact equality of the volume, which is suitable for hedging accounts, but
not for netting, where partial closure is possible if the volume of the transaction is less than the position
(and maybe more, in case of a reversal, but the DEAL_ENTRY_INOUT option is not here supported). All
these points should be finalized for real application.
Let's start the server node.exe wspubsub.j s and two copies of the Expert Advisor wstradecopier.mq5 on
different charts, in the same terminal. The usual scenario assumes that the Expert Advisor needs to be
launched on different accounts, but a "paradoxical" option is also suitable for checking the
performance: we will copy signals from one symbol to another.
In one copy of the Expert Advisor, we will leave the default settings, with the role of the publisher. It
should be placed on the EURUSD chart. In the second copy that runs on the GBPUSD chart, we change
the role to the subscriber. The string "EURUSD=GBPUSD" in the input parameter SymbolSubstitute
allows GBPUSD trading on EURUSD signals.
The connection data will be logged, with the HTTP headers and greetings we've already seen, so we'll
omit them.
Let's buy EURUSD and make sure that it is "duplicated" in the same volume for GBPUSD.
The following are fragments of the log (keep in mind that due to the fact that both Expert Advisors work
in the same copy of the terminal, transaction messages will be sent to both charts and therefore, to
facilitate the analysis of the log, you can alternately set the filters "EURUSD" and " USDUSD"):

---

## Page 1995

Part 7. Advanced language tools
1 995
7.8 Projects
(EURUSD,H1) TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, @ 0.99886, #=1461313378
(EURUSD,H1) DONE, D=1439023682, #=1461313378, V=0.01, @ 0.99886, Bid=0.99886, Ask=0.99886, Req=2
(EURUSD,H1) {"req" : {"a" : "TRADE_ACTION_DEAL", "s" : "EURUSD", "t" : "ORDER_TYPE_BUY", "v" : 0.01,
»   "f" : "ORDER_FILLING_FOK", "p" : 0.99886, "o" : 1461313378}, "res" : {"code" : 10009, "d" : 1439023682,
»   "o" : 1461313378, "v" : 0.01, "p" : 0.99886, "b" : 0.99886, "a" : 0.99886}, "deal" : {"d" : 1439023682,
»   "o" : 1461313378, "t" : "2022.09.19 16:45:50", "tmsc" : 1663605950086, "type" : "DEAL_TYPE_BUY",
»   "entry" : "DEAL_ENTRY_IN", "pid" : 1461313378, "r" : "DEAL_REASON_CLIENT", "v" : 0.01, "p" : 0.99886,
»   "s" : "EURUSD"}, "pos" : {"n" : 1}}
This shows the content of the executed request and its result, as well as a buffer with a JSON string
sent to the server.
Almost instantly, on the receiving side, on the GBPUSD chart, an alert is displayed with a message from
the server in a "raw" form and formatted after successful parsing in JsParser. In the "raw" form, the
"origin" property is stored, in which the server lets us know who is the source of the signal.

---

## Page 1996

Part 7. Advanced language tools
1 996
7.8 Projects
(GBPUSD,H1) Alert: {"origin":"publisher PUB_ID_001", "msg":{"req" : {"a" : "TRADE_ACTION_DEAL",
»   "s" : "EURUSD", "t" : "ORDER_TYPE_BUY", "v" : 0.01, "f" : "ORDER_FILLING_FOK", "p" : 0.99886,
»   "o" : 1461313378}, "res" : {"code" : 10009, "d" : 1439023682, "o" : 1461313378, "v" : 0.01,
»   "p" : 0.99886, "b" : 0.99886, "a" : 0.99886}, "deal" : {"d" : 1439023682, "o" : 1461313378,
»   "t" : "2022.09.19 16:45:50", "tmsc" : 1663605950086, "type" : "DEAL_TYPE_BUY",
»   "entry" : "DEAL_ENTRY_IN", "pid" : 1461313378, "r" : "DEAL_REASON_CLIENT", "v" : 0.01,
»   "p" : 0.99886, "s" : "EURUSD"}, "pos" : {"n" : 1}}}
(GBPUSD,H1){
(GBPUSD,H1)  req = 
(GBPUSD,H1)  {
(GBPUSD,H1)    a = TRADE_ACTION_DEAL
(GBPUSD,H1)    s = EURUSD
(GBPUSD,H1)    t = ORDER_TYPE_BUY
(GBPUSD,H1)    v =  0.01
(GBPUSD,H1)    f = ORDER_FILLING_FOK
(GBPUSD,H1)    p =  0.99886
(GBPUSD,H1)    o =  1461313378
(GBPUSD,H1)  }
(GBPUSD,H1)  res = 
(GBPUSD,H1)  {
(GBPUSD,H1)    code =  10009
(GBPUSD,H1)    d =  1439023682
(GBPUSD,H1)    o =  1461313378
(GBPUSD,H1)    v =  0.01
(GBPUSD,H1)    p =  0.99886
(GBPUSD,H1)    b =  0.99886
(GBPUSD,H1)    a =  0.99886
(GBPUSD,H1)  }
(GBPUSD,H1)  deal = 
(GBPUSD,H1)  {
(GBPUSD,H1)    d =  1439023682
(GBPUSD,H1)    o =  1461313378
(GBPUSD,H1)    t = 2022.09.19 16:45:50
(GBPUSD,H1)    tmsc =  1663605950086
(GBPUSD,H1)    type = DEAL_TYPE_BUY
(GBPUSD,H1)    entry = DEAL_ENTRY_IN
(GBPUSD,H1)    pid =  1461313378
(GBPUSD,H1)    r = DEAL_REASON_CLIENT
(GBPUSD,H1)    v =  0.01
(GBPUSD,H1)    p =  0.99886
(GBPUSD,H1)    s = EURUSD
(GBPUSD,H1)  }
(GBPUSD,H1)  pos = 
(GBPUSD,H1)  {
(GBPUSD,H1)    n =  1
(GBPUSD,H1)  }
(GBPUSD,H1)}
(GBPUSD,H1)Alert: Trade by subscription: market entry ORDER_TYPE_BUY 0.01 GBPUSD - Successful
The last of the above entries indicates a successful transaction on GBPUSD. On the trading tab of the
account, 2 positions should be displayed.
After some time, we close the EURUSD position, and the GBPUSD position should close automatically.

---

## Page 1997

Part 7. Advanced language tools
1 997
7.8 Projects
(EURUSD,H1) TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_SELL, V=0.01, ORDER_FILLING_FOK, @ 0.99881, #=1461315206, P=1461313378
(EURUSD,H1) DONE, D=1439025490, #=1461315206, V=0.01, @ 0.99881, Bid=0.99881, Ask=0.99881, Req=4
(EURUSD,H1) {"req" : {"a" : "TRADE_ACTION_DEAL", "s" : "EURUSD", "t" : "ORDER_TYPE_SELL", "v" : 0.01,
»   "f" : "ORDER_FILLING_FOK", "p" : 0.99881, "o" : 1461315206, "q" : 1461313378}, "res" : {"code" : 10009,
»   "d" : 1439025490, "o" : 1461315206, "v" : 0.01, "p" : 0.99881, "b" : 0.99881, "a" : 0.99881},
»   "deal" : {"d" : 1439025490, "o" : 1461315206, "t" : "2022.09.19 16:46:52", "tmsc" : 1663606012990,
»   "type" : "DEAL_TYPE_SELL", "entry" : "DEAL_ENTRY_OUT", "pid" : 1461313378, "r" : "DEAL_REASON_CLIENT",
»   "v" : 0.01, "p" : 0.99881, "m" : -0.05, "s" : "EURUSD"}, "pos" : {"n" : 0}}
If the deal had a type DEAL_ENTRY_IN for the first time, now it is DEAL_ENTRY_OUT. The alert
confirms the receipt of the message and the successful closing of the duplicate position.
(GBPUSD,H1) Alert: {"origin":"publisher PUB_ID_001", "msg":{"req" : {"a" : "TRADE_ACTION_DEAL",
»   "s" : "EURUSD", "t" : "ORDER_TYPE_SELL", "v" : 0.01, "f" : "ORDER_FILLING_FOK", "p" : 0.99881,
»   "o" : 1461315206, "q" : 1461313378}, "res" : {"code" : 10009, "d" : 1439025490, "o" : 1461315206,
»   "v" : 0.01, "p" : 0.99881, "b" : 0.99881, "a" : 0.99881}, "deal" : {"d" : 1439025490,
»   "o" : 1461315206, "t" : "2022.09.19 16:46:52", "tmsc" : 1663606012990, "type" : "DEAL_TYPE_SELL",
»   "entry" : "DEAL_ENTRY_OUT", "pid" : 1461313378, "r" : "DEAL_REASON_CLIENT", "v" : 0.01,
»   "p" : 0.99881, "m" : -0.05, "s" : "EURUSD"}, "pos" : {"n" : 0}}}
...
(GBPUSD,H1)Alert: Trade by subscription: market exit ORDER_TYPE_SELL 0.01 GBPUSD - Successful
Finally, next to the Expert Advisor wstradecopier.mq5, we create a project file wstradecopier.mqproj  to
add a description and necessary server files to it (in the old directory
MQL5/Experts/p7/MQL5Book/Web/).
To summarize: we have organized a technically extensible, multi-user system for exchanging trading
information via a socket server. Due to the technical features of web sockets (permanent open
connection), this implementation of the signal service is more suitable for short-term and high-
frequency trading, as well as for controlling arbitrage situations in quotes.
Solving the problem required combining several programs on different platforms and connecting a large
number of dependencies, which is what usually characterizes the transition to the project level. The
development environment is also expanded, going beyond the compiler and source code editor. In
particular, the presence in the project of the client or server parts usually involves the work of different
programmers responsible for them. In this case, shared projects in the cloud and with version control
become indispensable.
Please note that when developing a project in the folder MQL5/Shared Proj ects via MetaEditor, header
files from the standard directory MQL5/Include are not included in the shared storage. On the other
hand, creating a dedicated folder Include inside your project and transferring the necessary standard
mqh files to it will lead to duplication of information and potential discrepancies in the versions of
header files. This behavior is likely to be improved in MetaEditor.
Another point for public projects is the need to administer users and authorize them. In our last
example, this issue was only identified but not implemented. However, the mql5.com site provides a
ready-made solution based on the well-known OAuth protocol. Anyone who has an mql5.com account
can get familiar with the principle of OAuth and configure it for their web service: just find the section
Applications (link looking like https://www.mql5.com/en/users/<login> /apps) in your profile. By
registering a web service in mql5.com applications, you will be able to authorize users through the
mql5.com website.

---

## Page 1998

Part 7. Advanced language tools
1 998
7.9 Native python support
7.9 Native python support
The potential success of automated trading largely depends on the breadth of technology that is
available in the implementation of the idea. As we have already seen in the previous sections, MQL5
allows you to go beyond strictly applied trading tasks and provides opportunities for integration with
external services (for example, based on network functions and custom symbols), processing and
storing data using relational databases, as well as connecting arbitrary libraries.
The last point allows you to ensure interaction with any software that provides API in the DLL format.
Some developers use this method to connect to industrial distributed DBMSs (instead of the built-in
SQLite), math packages like R or MATLAB, and other programming languages.
Python has become one of the most popular programming languages. Its feature is a compact core,
which is complemented by packages which are ready-made collections of scripts for building application
solutions. Traders benefit from the wide selection and functionality of the packages for fundamental
market analysis (statistical calculations, data visualization) and testing of trading hypotheses, including
machine learning.
Following this trend, MQ introduced Python support in MQL5 in 201 9. This tighter "out-of-the-box"
integration allows the complete transfer of technical analysis and trading algorithms to the Python
environment.
From a technical point of view, integration is achieved by installing the "MetaTrader5" package in
Python, which organizes interprocess interaction with the terminal (at the time of writing this, through
the ipykernel/RPC mechanism).
Among the functions of the package, there are full analogs of the built-in MQL5 functions for obtaining
information about the terminal, trading account, symbols in Market Watch, quotes, ticks, Depth of
Market, orders, positions, and deals. In addition, the package allows you to switch trading accounts,
send trade orders, check margin requirements, and evaluate potential profits/losses in real-time.
However, integration with Python has some limitations. In particular, it is not possible in Python to
implement event handling such as OnTick, OnBookEvent, and others. Because of this, it is necessary to
use an infinite loop to check new prices, much like we were forced to do in MQL5 scripts. The analysis
of the execution of trade orders is just as difficult: in the absence of OnTradeTransaction, more code
would be needed to know if a position was fully or partially closed. To bypass these restrictions, you
can organize the interaction of the Python script and MQL5, for example, through sockets. The
mql5.com site features articles with examples of the implementation of such a bridge.
Thus, it seems that it is only natural to use Python in conjunction with MetaTrader 5 for machine
learning tasks that deal with quotes, ticks, or trading account history. Unfortunately, you can't get
indicator readings in Python.
7.9.1  Installing Python and the MetaTrader5 package
To study the materials in this chapter, Python must be installed on your computer. If you haven't
installed it yet, download the latest version of Python (e.g. 3.1 0 at the time of writing) from
https://www.python.org/downloads/windows.
When installing Python, it is recommended to check the "Add Python to PATH" flag so that you can run
Python scripts from the command line from any folder.

---

## Page 1999

Part 7. Advanced language tools
1 999
7.9 Native python support
Once Python is downloaded and running, install the MetaTrader5 module from the command line (here
pip is a standard Python package manager program):
pip install MetaTrader5
Subsequently, you can check the package update with the following command line:
pip install --upgrade MetaTrader5
The syntax for adding other commonly used packages is similar. In particular, many scripts require
data analysis and visualization packages: pandas and matplotlib, respectively.
pip install matplotlib
pip install pandas
You can create a new Python script directly from the MQL5 Wizard in MetaEditor. In addition to the
script name, the user can select options for importing multiple packages, such as TensorFlow, NumPy,
or Datetime.
Scripts by default are suggested to be placed in the folder MQL5/Scripts. Newly created and existing
Python scripts are displayed in the MetaTrader 5 Navigator, marked with a special icon, and can be
launched from the Navigator in the usual way. Python scripts can be executed on the chart in parallel
with other MQL5 scripts and Expert Advisors. To stop a script if its execution is looped, simply remove it
from the chart.
Running Python script in the terminal
The Python script launched from the terminal receives the name of the symbol and the timeframe of
the chart through command line parameters. For example, we can run the following script on the
EURUSD, H1  chart, in which the arguments are available as the sys.argv array:

---

## Page 2000

Part 7. Advanced language tools
2000
7.9 Native python support
import sys
   
print('The command line arguments are:')
for i in sys.argv:
   print(i)
It will output to the expert log:
The command line arguments are:
C:\Program Files\MetaTrader 5\MQL5\Scripts\MQL5Book\Python\python-args.py
EURUSD
60
In addition, a Python script can be run directly from MetaEditor by specifying the Python installation
location in the editor Settings dialog, tab Compilers – then the compilation command for files with the
extension *.py becomes a run command.
Finally, Python scripts can also be run in their native environment by passing them as parameters in
python.exe calls from the command line or from another IDE (Integrated Development Environment)
adapted for Python, such as Jupyter Notebook.
If algorithmic trading is enabled in the terminal, then trading from Python is also enabled by default. To
further protect accounts when using third-party Python libraries, the platform settings provide the
option "Disable automatic trading via external Python API". Thus, Python scripts can selectively block
trading, leaving it available to MQL programs. When this option is enabled, trading function calls in a
Python script will return error 1 0027 (TRADE_RETCODE_CLIENT_DISABLES_AT) indicating that
algorithmic trading is disabled by the client terminal.
MQL5 vs Python
 Python is an interpreted language, unlike compiled MQL5. For us as developers, this makes life
easier because we don't need a separate compilation phase to get a working program. However,
the execution speed of scripts in Python is noticeably lower than those compiled in MQL5. 
Python is a dynamically typed language: the type of a variable is determined by the value we put in
it. On the one hand, this gives flexibility, but it also requires caution in order to avoid unforeseen
errors. MQL5 uses static typing, that is, when describing variables, we must explicitly specify their
type, and the compiler monitors type compatibility. 
Python itself "cleans the garbage", that is, frees the memory allocated by the application program
for objects. In MQL5 we have to follow up the timely call of delete for dynamic objects. 
In Python syntax, source code indentation plays an important role. If you need to write a compound
statement (for example, a loop or conditional) with a block of several nested statements, then
Python uses spaces or tabs for this purpose (they must be equal in size within the block). Mixing
tabs and spaces is not allowed. The wrong indentation will result in an error. In MQL5, we form
blocks of compound statements by enclosing them in curly brackets { ... }, but formatting does not
play a role, and you can apply any style you like without breaking the program's performance. 
Python functions support two types of parameters: named and positional. The second type
corresponds to what we are used to in MQL5: the value for each parameter must be passed strictly
in its order in the list of arguments (according to the function prototype). In contrast, named
parameters are passed as a combination of name and value (with '=' between them), and therefore
they can be specified in any order, for example, func(param2 = value2, param1  = value1 ).

---

