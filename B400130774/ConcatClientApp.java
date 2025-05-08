import java.rmi.Naming;
import java.util.Scanner;

// Same interface as used in server
interface ConcatService extends java.rmi.Remote {
    String concatenate(String str1, String str2) throws java.rmi.RemoteException;
}

public class ConcatClientApp {
    public static void main(String[] args) {
        try {
            ConcatService stub = (ConcatService) Naming.lookup("rmi://localhost:5000/concat");

            Scanner sc = new Scanner(System.in);
            System.out.print("Enter first string: ");
            String str1 = sc.nextLine();
            System.out.print("Enter second string: ");
            String str2 = sc.nextLine();

            String result = stub.concatenate(str1, str2);
            System.out.println("Concatenated string: " + result);
        } catch (Exception e) {
            System.out.println("Client error: " + e);
        }
    }
}












































































































// 1. Code Solves/Satisfies the Problem Statement:

// Yes, the code effectively implements a distributed application using RMI.

// The server-side code defines a ConcatService interface and its implementation ConcatServerApp, which provides the remote concatenate method.

// The client-side code looks up the remote object and calls the concatenate method to get the desired result.

// 2. Output Correct or Not:

// Yes, the output is correct.  The program correctly concatenates "hello" and "world" to produce "helloworld".

// 3. Why the Output is Like That and What it Represents:

// The server binds the ConcatServerApp object to the RMI registry with the name "concat".

// The client retrieves this remote object from the registry.

// The client then prompts the user for two strings, "hello" and "world".

// These strings are passed to the concatenate method of the remote object.

// The server executes the concatenate method, which concatenates the two strings.

// The server returns the concatenated string "helloworld" to the client.

// The client receives the result and prints it to the console.

// 4. Line-by-Line Explanation (Short):

// Server-Side:

// import java.rmi.*; etc.: Imports necessary RMI classes.

// interface ConcatService extends Remote: Defines the remote interface with the concatenate method.

// public class ConcatServerApp extends UnicastRemoteObject implements ConcatService: Implements the remote interface.

// protected ConcatServerApp() throws RemoteException { super(); }: Constructor for the server object.

// public String concatenate(String str1, String str2) throws RemoteException:  The implementation of the remote method.

// LocateRegistry.createRegistry(5000);: Creates the RMI registry on port 5000.

// ConcatServerApp obj = new ConcatServerApp();: Creates an instance of the server object.

// Naming.rebind("rmi://localhost:5000/concat", obj);: Binds the server object to the registry with a name.

// System.out.println("Server started...");: Prints a message.

// Client-Side:

// import java.rmi.Naming; etc.: Imports necessary RMI classes.

// interface ConcatService extends java.rmi.Remote:  The client also needs the interface.

// ConcatService stub = (ConcatService) Naming.lookup("rmi://localhost:5000/concat");: Looks up the remote object from the registry.

// Scanner sc = new Scanner(System.in);: Creates a Scanner to read input.

// System.out.print("Enter first string: ");...: Prompts for and reads the first string.

// System.out.print("Enter second string: ");...: Prompts for and reads the second string.

// String result = stub.concatenate(str1, str2);: Calls the remote method.

// System.out.println("Concatenated string: " + result);: Prints the result.

// 5. Potential Oral Questions:

// Here are some potential questions your examiner might ask:

// General RMI Concepts:

// What is RMI? How does it work? How is it different from RPC?

// Explain the client-server architecture in RMI.

// What is a remote interface? What is its role?

// What is a stub and a skeleton? (Though skeletons are mostly hidden in modern RMI)

// What is the purpose of the RMI registry?

// What are the advantages and disadvantages of using RMI?

// Code Details:

// Explain the ConcatService interface. Why does both client and server need it?

// Explain the ConcatServerApp class. What does UnicastRemoteObject do?

// What is the purpose of Naming.lookup() and Naming.rebind()?

// What is the role of the port number (5000) in this code?

// What happens if the server is not running when the client tries to connect?

// RMI Registry:

// What is the RMI registry and why is it needed?

// Can you have an RMI application without a registry? (Yes, with some extra work)

// How does the client find the remote object?

// Error Handling:

// How are exceptions handled in RMI?  What is RemoteException?

// Distributed Systems:

// How does RMI facilitate distributed computing?

// What are some challenges in distributed systems that RMI addresses?