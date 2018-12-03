import React from 'react';
import { ScrollView, StyleSheet, View, TextInput, Button } from 'react-native';
import { QuestionList } from '../components/Questions/QuestionList';

export default class NewDocumentScreen extends React.Component {
  static navigationOptions = {
    title: 'New quiz',
  };

  state = {
    text: "",
    docTitle: ""
  }

  getAnswers = async(questions) => {
    let answers = questions.map((item) => [0, 1, 2, 3, 4].map((num) => ["Answer", String(num), "for", item].join(' ')));
    return answers;
  }

  getQuestions = async() => {

    let questions = this.state.text.split(" ");
    console.log(questions);
    questions = questions.map((item) => ["What is", item, "?"].join(' '));


    let answers = await this.getAnswers(questions);

    console.log(questions);
    console.log(answers);

    list = [];

    for (i = 0; i < questions.length; i++) {

      let toAppend = {
        key: String(i),
        question: questions[i],
        choices: [
          { key: '0', text: "ans1", color: 'green'},
          { key: '1', text: "ans2", color: 'red'},
          { key: '2', text: "ans3", color: 'red' },
        ]
      }

      list.push(toAppend);
    }



    this.props.navigation.navigate('NewQuiz', {
      text: this.state.text,
      docTitle: this.state.docTitle,
      list: list,
    });
  }

  // <TextInput
  //   style={styles.docTitle}
  //   placeholder="Enter quiz title here!"
  //   value={this.state.docTitle}
  //   onChangeText{(text) => this.setState({docTitle: text})} />

  render() {
    return (
      <View style={styles.container}>
        <TextInput
          style={styles.docTitle}
          placeholder="Enter the title of your quiz here"
          onChangeText={(text) => this.setState({docTitle: text})}
          value={this.state.docTitle}
        />
        <TextInput
          style={styles.docInput}
          placeholder="Enter your document text here!"
          onChangeText={(text) => this.setState({text})}
        />
        <Button style={styles.makeQuizButton} title="Make me a quiz" onPress={this.getQuestions} />
      </View>
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
  docTitle: {
    flex: 1,
    marginHorizontal: 10,
  },
  docInput: {
    flex: 9,
    backgroundColor: '#ddd',
    marginHorizontal: 10,
  },
  makeQuizButton: {
  },
});
