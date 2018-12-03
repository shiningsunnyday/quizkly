import React from 'react';
import { AsyncStorage, TextInput, Button, StyleSheet, View, Text } from 'react-native';

export default class LoadingScreen extends React.Component {

  constructor(props) {
    super(props);
    this._getAsync();
  }

  componentDidMount() {
    this.props.navigation.navigate('Main');
  }

  state = {
    text: "Welcome back! Please re-enter your.",
    buttonText: "Log in",
    username: "",
    password: "",
    isNewUser: false,
  }

  _signUp = async () => {

    try {
      await AsyncStorage.setItem('username', this.state.username);
    } catch (error) {
      console.log("Cant set username");
    }
    await AsyncStorage.setItem('password', this.state.password);

    console.log("Signed up!");
    props.reload();



  }

  _logIn = async () => {

    if(this.state.isNewUser) {
      this._signUp();
      console.log("Signed up");
    } else {
      let correctUsername = await AsyncStorage.getItem('username');
      let correctPassword = await AsyncStorage.getItem('password');


      if(correctUsername === this.state.username && correctPassword === this.state.password) {
        this.props.navigation.navigate('Main');
      }

    }


  }

  _getAsync = async () => {

    const userToken = await AsyncStorage.getItem('username') || 'none';
    if(userToken == 'none') {
      this.setState({
        text: "Hello! Please sign up.",
        isNewUser: true,
        buttonText: "Sign up",
      });
    }

  }

  render() {
    return (
      <View style={styles.container}>
        <View style={styles.title}>
          <Text style={styles.titleText}>{this.state.text}</Text>
        </View>
        <TextInput
          style={styles.textInput}
          onChangeText={(username) => this.setState({
            username: username,
          })}
          placeholder="Username"
          value={this.state.username} />
        <TextInput
          style={styles.textInput}
          onChangeText={(password) => this.setState({
            password: password
          })}
          placeholder="Password"
          value={this.state.password} />
        <Button title={this.state.buttonText} onPress={this._logIn} />
      </View>
    )
  }




}

const styles = StyleSheet.create({
  title: {
    margin: 20,
    marginTop: 40,
    backgroundColor: '#eee',
  },
  titleText: {
    textAlign: 'center',
    fontSize: 20,
  },
  container: {
    flex: 1,
    backgroundColor: '#ddd',
  },
  textInput: {
    margin: 20,
    marginTop: 40,
    height: 40,
    borderColor: 'gray',
    borderWidth: 1,
  },
})
