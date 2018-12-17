import React from 'react';
import { AsyncStorage, ScrollView, Text, StyleSheet, View, TextInput, Button } from 'react-native';
import { QuestionList } from '../components/Questions/QuestionList';

export default class NewQuizScreen extends React.Component {
  static navigationOptions = {
    title: 'Your quiz',
  };

  constructor(props) {
    super(props);
  }


  componentDidMount() {

  }

  startTest = async() => {

    console.log("I'm about to start a new test!");
    this.props.navigation.navigate('Test', {
      params: this.props.navigation.state.params
    });

  }

  saveList = async() => {

    const params = this.props.navigation.state.params;


    let currentQuizzes = await AsyncStorage.getItem('quizzes') || [];


    if(currentQuizzes.length > 0) {
      currentQuizzes = JSON.parse(currentQuizzes);
      console.log("Current length is", currentQuizzes.length);
    }

    currentQuizzes.push({
      docTitle: params.docTitle,
      text: params.text,
      list: params.list,
    });

    console.log("After saving list, length is", currentQuizzes.length);

    await AsyncStorage.setItem('quizzes', JSON.stringify(currentQuizzes));

    this.props.navigation.navigate('Docs');

  }

  state = {
  }

  // <TextInput
  //   style={styles.docTitle}
  //   placeholder="Enter quiz title here!"
  //   value={this.state.docTitle}
  //   onChangeText{(text) => this.setState({docTitle: text})} />

  render() {
    const params = this.props.navigation.state.params;
    console.log(params);
    return (
      <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
        <QuestionList startTest={this.startTest} saveList={this.saveList} list={params.list}/>
      </ScrollView>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 15,
    backgroundColor: '#eee',
    flexDirection: 'column',
  },
  contentContainer: {
    paddingTop: 20,
    flex: 1,
    paddingBottom: 20,
  },
});
